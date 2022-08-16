import json
from pathlib import Path, PureWindowsPath
import torch
import sys
import os
import wandb
from utils.cli import parse_args
from utils.utils import create_loss_weight
from models.baseline import Baseline
from utils.seqlab import preprocessing_kmer
from transformers import BertTokenizer

def train(model, optimizer, scheduler, train_dataloader, eval_dataloader, batch_size, num_epochs, device, save_dir, wandb, start_epoch=0, device_list=[], loss_weight=None):
    model.to(device)
    num_labels = model.num_labels
    if len(device_list) > 0:
        torch.nn.DataParallel(device_list)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "latest")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if loss_weight:
        loss_weight = loss_weight.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)

    wandb.define_metric("epoch")
    wandb.define_metric("training/epoch_loss", step_metric="epoch")
    wandb.define_metric("validation/loss", step_metric="epoch")
    wandb.define_metric("validation/accuracy", step_metric="epoch")

    training_log_path = os.path.join(save_dir, "training_log.csv")
    training_log = open(training_log_path, "x")
    training_log.write("epoch,step,loss\n")
    validation_log_path = os.path.join(save_dir, "validation_log.csv")
    validation_log = open(validation_log_path, "x")
    validation_log.write("epoch,step,input,prediction,target,loss\n")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, token_type_ids, target_labels = tuple(t.to(device) for t in batch)
            with torch.cuda.amp.auto_grad():
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
        for step, batch in enumerate(eval_dataloader):
            input_ids, attention_mask, token_type_ids, target_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                predictions = model(input_ids)
                loss = criterion(predictions.view(-1, num_labels), target_labels.view(-1))
            for input_id, pred, label in zip(input_ids, predictions, target_labels):
                q = input_id[1:] # Remove CLS token from input
                q = [a for a in q if a >= 0] # Remove padding token.
                p = pred[1:] # Remove CLS token from prediction.
                p = p[0:len(q)] # Remove padding prediction. 
                pval, p = torch.max(p, 1)               
                l = label[1:] # Remove CLS token from label
                l = l[0:len(q)] # Remove padding label.
                accuracy = 0
                for i, j in zip(p, l):
                    accuracy += (1 if i == j else 0)
                accuracy = accuracy / len(q) * 100
                qlist = q.tolist()
                qlist = [str(a) for a in qlist]
                qlist = " ".join(qlist)
                plist = p.tolist()
                plist = [str(a) for a in plist]
                plist = " ".join(plist)
                llist = l.tolist()
                llist = [str(a) for a in llist]
                llist = " ".join(llist)
                validation_log.write(f"{epoch},{step},{qlist},{plist},{llist}\n")
            
        training_log.close()
        validation_log.close()
        torch.save({
            "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }, checkpoint_dir)
            

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    training_config_path = args.get("training-config", False)
    device = args.get("device", False)
    resume_run_ids = args.get("resume-run-ids", False)
    model_config_dir = args.get("model-config-dir", False)
    model_config_names = args.get("model-config-names", False)
    batch_size = args.get("batch-size", 1)
    num_epochs = args.get("num-epochs", 1)
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.01)
    training_config = json.load(open(training_config_path, "r"))
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
