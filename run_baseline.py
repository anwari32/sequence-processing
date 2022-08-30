import json
from pathlib import Path, PureWindowsPath
import torch
import sys
import os
import wandb
from utils.cli import parse_args
from utils.utils import create_loss_weight
from models.baseline import Baseline
from utils.seqlab import Index_Dictionary, preprocessing_kmer
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from utils.metrics import Metrics
from utils.seqlab import NUM_LABELS

def train(model, optimizer, scheduler, train_dataloader, eval_dataloader, batch_size, num_epochs, device, save_dir, wandb, start_epoch=0, device_list=[], loss_weight=None):
    model.to(device)
    num_labels = model.num_labels
    if len(device_list) > 0:
        torch.nn.DataParallel(device_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "latest")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if loss_weight != None:
        loss_weight = loss_weight.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)

    wandb.define_metric("epoch")
    wandb.define_metric("training/epoch_loss", step_metric="epoch")
    wandb.define_metric("validation/loss", step_metric="epoch")
    wandb.define_metric("validation/accuracy", step_metric="epoch")

    training_log_path = os.path.join(save_dir, "training_log.csv")
    training_log = open(training_log_path, "x")
    training_log.write("epoch,step,loss\n")
    
    len_train_dataloader = len(train_dataloader)
    len_eval_dataloader = len(eval_dataloader)
    for label_index in range(NUM_LABELS):
        label = Index_Dictionary[label_index]
        wandb.define_metric(f"validation/{label}", step_metric="epoch")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), total=len_train_dataloader, desc=f"Training {epoch + 1}/{num_epochs}"):
            input_ids, attention_mask, token_type_ids, target_labels = tuple(t.to(device) for t in batch)
            with torch.cuda.amp.autocast():
                input_ids = input_ids.reshape(input_ids.shape[0], input_ids.shape[1], 1)
                input_ids = input_ids.float()
                pred = model(input_ids)
                loss = criterion(pred.view(-1, num_labels), target_labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"loss": loss.item()})
            epoch_loss += loss.item()
            training_log.write(f"{epoch},{step},{loss.item()}\n")
        wandb.log({"training/epoch_loss": epoch_loss, "epoch": epoch})
        scheduler.step()

        model.eval()
        cf_metric_name = f"cf_matrix/confusion-matrix-{wandb.run.name}-{epoch}"
        wandb.define_metric(cf_metric_name, step_metric="epoch")
        validation_log_path = os.path.join(save_dir, f"validation_log.{epoch}.csv")
        if os.path.exists(validation_log_path):
            os.remove(validation_log_path)
        validation_log = open(validation_log_path, "x")
        validation_log.write("epoch,step,input,prediction,target,loss,accuracy\n")
        y_test = []
        y_pred = []
        for step, batch in tqdm(enumerate(eval_dataloader), total=len_eval_dataloader, desc=f"Validation {epoch + 1}/{num_epochs}"):
            input_ids, attention_mask, token_type_ids, target_labels = tuple(t.to(device) for t in batch)
            input_ids = input_ids.reshape(input_ids.shape[0], input_ids.shape[1], 1)
            input_ids = input_ids.float()
            with torch.no_grad():
                predictions = model(input_ids)
                loss = criterion(predictions.view(-1, num_labels), target_labels.view(-1))
            for input_id, pred, label in zip(input_ids, predictions, target_labels):
                q = input_id.view(-1).tolist()
                qlist = [str(int(a)) for a in q]
                qlist = " ".join(qlist)

                pval, p = torch.max(pred, 1)
                p = p.tolist()
                plist = [str(a) for a in p]
                plist = " ".join(plist)

                l = label.tolist()
                llist = [str(a) for a in l]
                llist = " ".join(llist)
                
                filtered_label = l[1:] # Remove CLS token.
                filtered_label = [a for a in filtered_label if a >= 0] # Remove special tokens.
                filtered_pred = p[1:] # Remove CLS token.
                filtered_pred = p[0:len(filtered_label)] # Remove special tokens.
                y_test = np.concatenate((y_test, filtered_label))
                y_pred = np.concatenate((y_pred, filtered_pred))
                
                # metrics
                accuracy = 0
                for i, j in zip(p, l):
                    accuracy += (1 if i == j else 0)
                accuracy = accuracy / len(q) * 100
                validation_log.write(f"{epoch},{step},{qlist},{plist},{llist},{accuracy},{loss.item()}\n")
                wandb.log({"validation/accuracy": accuracy, "epoch": epoch})

        metrics = Metrics(y_pred, y_test)
        metrics.calculate()
        for label_index in range(NUM_LABELS):
            label = Index_Dictionary[label_index]
            wandb.log({
                f"validation/{label}": metrics.precission(label_index, percentage=True),
                "epoch": epoch
            })

        wandb.log({
            cf_metric_name: wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test, 
                preds=y_pred,
                class_names=[c for c in range(NUM_LABELS)])
            })
        validation_log.close()
        wandb.save(validation_log_path)

        torch.save({
            "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }, os.path.join(checkpoint_dir, "checkpoint.pth"))
    
    training_log.close()


            
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    training_config_path = args.get("training-config", False)
    device = args.get("device", False)
    resume_run_ids = args.get("resume-run-ids", False)
    model_config_dir = args.get("model-config-dir", False)
    model_config_names = args.get("model-config-names", False)
    run_name = args.get("run-name", "baseline")
    device_list = args.get("device-list", [])
    project_name = args.get("project-name", "baseline")
    use_weighted_loss = args.get("use-weighted-loss", False)

    model = Baseline()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)
    training_config = json.load(open(training_config_path, "r"))
    batch_size = args.get("batch-size", training_config.get("batch_size", 1))
    num_epochs = args.get("num-epochs", training_config.get("num_epochs_size", 1))
    training_data_path = training_config.get("train_data", False)
    if training_data_path:
        training_data_path = str(Path(PureWindowsPath(training_data_path)))
    validation_data_path = training_config.get("validation_data", False)
    if validation_data_path:
        validation_data_path = str(Path(PureWindowsPath(validation_data_path)))
    pretrained_path = str(Path(PureWindowsPath(training_config.get("pretrained", os.path.join("pretrained", "3-new-12w-0")))))
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    train_dataloader, eval_dataloader = preprocessing_kmer(training_data_path, tokenizer, batch_size), preprocessing_kmer(validation_data_path, tokenizer, batch_size)
    loss_weight = create_loss_weight(training_data_path) if use_weighted_loss else None
    
    os.environ["WANDB_MODE"] = "offline" if args.get("offline", False) else "online"
    run_id = args.get("resume-run-ids")[0] if args.get("resume-run-ids") else wandb.util.generate_id()
    runname = f"{run_name}-{run_id}"
    save_dir = os.path.join("run", runname)
    checkpoint_dir = os.path.join(save_dir, "latest")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    run = wandb.init(id=run_id, project=project_name, name=runname, reinit=True, resume="allow")
    start_epoch = 0
    if run.resumed:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint.get("model")))
        optimizer.load_state_dict(torch.load(checkpoint.get("optimizer")))
        scheduler.load_state_dict(torch.load(checkpoint.get("scheduler")))
        epoch = int(checkpoint.get("epoch"))
        start_epoch = epoch + 1

    print("Running Baseline")
    print(f"Scenario {run.name}")
    print(f"Num epocs {num_epochs}")
    print(f"Batch size {batch_size}")
    print(f"Start epoch {start_epoch}")
    print(f"Save Dir {save_dir}")
    str_device_list = [torch.cuda.get_device_name(d) for d in device_list]
    str_device_list = ', '.join(str_device_list)
    print(f"Device(s) {torch.cuda.get_device_name(device)} {str_device_list}")
    str_loss_weight = loss_weight if use_weighted_loss else ""
    print(f"Use weighted loss {use_weighted_loss} {str_loss_weight}")

    train(model, 
        optimizer, 
        scheduler,
        train_dataloader,
        eval_dataloader,
        batch_size,
        num_epochs,
        device,
        save_dir,
        wandb,
        start_epoch,
        device_list,
        loss_weight)

    run.finish()
