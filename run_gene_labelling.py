from multiprocessing.sharedctypes import Value
import os
from getopt import getopt
from re import L
from tabnanny import check
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast
from torch.optim import AdamW, lr_scheduler
import torch
import wandb
import sys
from datetime import datetime
import json
from transformers import BertForMaskedLM, BertTokenizer
from utils.seqlab import preprocessing_gene_kmer
from tqdm import tqdm
import pathlib
from models.genlab import DNABERT_RNN

def parse_args(argvs):
    opts, args = getopt(argvs, "m:t:c:d:r:p:w:", [
        "training-config=", "model-config=", "model-config-dir=", "device=", "device-list=", "run-name=", "project-name=", "batch-size=", "num-epochs=", "resume-run-id=", "loss-weight="
    ])
    output = {}
    for o, a in opts:
        if o in ["-t", "--training-config"]:
            output["training-config"] = a
        elif o in ["-d", "--device"]:
            output["device"] = a
        elif o in ["--device-list"]:
            output["device-list"] = a.split(",")
        elif o in ["-m", "--model-config-dir"]:
            output["model-config-dir"] = a
        elif o in ["-c", "--model-config"]:
            output["model-config"] = a.split(",")
        elif o in ["--run-name"]:
            output["run-name"] = a
        elif o in ["--project-name"]:
            output["project-name"] = a
        elif o in ["--batch-size"]:
            output["batch-size"] = int(a)
        elif o in ["--num-epochs"]:
            output["num-epochs"]= int(a)
        elif o in ["--resume-run-id"]:
            output["resume-run-id"] = a
        elif o in ["-w", "--loss-weight"]:
            output["loss-weight"] = a
        else:
            raise ValueError(f"Argument {o} not recognized.")
    return output

def train(model, optimizer, scheduler, train_dataloader, validation_dataloader, num_epochs, device, save_dir, wandb, start_epoch=0, device_list=[], criterion_weight=None):
    model.to(device)
    num_labels = model.num_labels

    loss_weight_10 = torch.Tensor([0.0001555761504390511, 0.9998775560181217, 0.0001555761504390511, 0.9969478696129899, 1.0, 0.9971913542557089, 0.0001555761504390511, 0.0019914280314346157])
    loss_weight_25 = torch.Tensor([0.00015717190553216775, 0.9999030960802364, 0.00015717190553216775, 0.9997093445720099, 1.0, 1.0, 0.00015717190553216775, 0.0020191783293449415])
    loss_weight = torch.Tensor([0.0001583913443766686, 1.0, 0.0001583913443766686, 1.0, 0.9999639301688068, 0.9999639301688068, 0.0001583913443766686, 0.0020260445716651057])

    criterion = None
    if loss_weight == "10":
        criterion = CrossEntropyLoss(weight=loss_weight_10)
    elif loss_weight == "25":
        criterion = CrossEntropyLoss(weight=loss_weight_25)
    elif loss_weight == "100":
        criterion = CrossEntropyLoss(weight=loss_weight)
    else:
        criterion = CrossEntropyLoss(weight=None)
    
    n_train_data = len(train_dataloader)
    n_validation_data = len(validation_dataloader)
    training_log_path = os.path.join(save_dir, "training_log.csv")
    validation_log_path = os.path.join(save_dir, "validation_log.csv")
    for p in [training_log_path, validation_log_path]:
        if os.path.exists(p):
            os.remove(p)
    training_log = open(training_log_path, "x")
    validation_log = open(validation_log_path, "x")
    training_log.write("epoch,step,sequence,prediction,target,loss\n")
    validation_log.write("epoch,step,sequence,prediction,target,loss,accuracy\n")

    wandb.define_metric("epoch")
    wandb.define_metric("accuracy", step_metric="epoch")
    wandb.define_metric("epoch_loss", step_metric="epoch")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        hidden_units = None
        model.train()
        mark = None
        for step, batch in tqdm(enumerate(train_dataloader), total=n_train_data, desc=f"Training {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids, labels, markers = tuple(t.to(device) for t in batch)
            if mark == None:
                mark = markers
            if mark != markers:
                hidden_units = None
                mark = markers
            with autocast():
                prediction, hidden_units = model(input_ids, attention_mask, hidden_units)
                batch_loss = criterion(prediction.view(-1, num_labels), labels.view(-1))
            batch_loss.backward()
            wandb.log({"train/batch_loss": batch_loss.item(), "learning_rate": scheduler.optimizer.param_groups[0]['lr']})
            optimizer.step()                
            for inputs, pred, label, m in zip(input_ids, prediction, labels, markers):
                pred_vals, pred_indices = torch.max(pred, 1)
                pred_list = " ".join([str(a) for a in pred_indices.tolist()])
                label_list = " ".join([str(a) for a in label.tolist()])
                input_list = " ".join([str(a) for a in inputs.tolist()])
                loss = criterion(pred, label)
                training_log.write(f"{epoch},{step},{input_list},{pred_list},{label_list},{loss.item()}\n")
                wandb.log({
                    "train/loss": loss.item(),
                    "marker": m.item()
                })
            epoch_loss += batch_loss.item()
        
        wandb.log({"epoch": epoch, "epoch_loss": epoch_loss})
        scheduler.step()

        model.eval()
        hidden_units = None
        mark = None
        average_accuracy = 0
        for step, batch in tqdm(enumerate(validation_dataloader), total=n_validation_data, desc=f"Validating {epoch + 1}/{num_epochs}"):
            input_ids, attention_mask, token_type_ids, labels, markers = tuple(t.to(device) for t in batch)
            if mark == None:
                mark = markers
            if mark != markers:
                hidden_units = None
                mark = markers
            with torch.no_grad():
                prediction, hidden_units = model(input_ids, attention_mask, hidden_units)
                batch_loss = criterion(prediction.view(-1, num_labels), labels.view(-1))
            wandb.log({"validation/batch_loss": batch_loss.item()})
            
            for input, pred, label in zip(input_ids, prediction, labels):
                pred_vals, pred_indices = torch.max(pred, 1)
                pred_indices = pred_indices.tolist()
                label_indices = label.tolist()
                pred_list = " ".join([str(a) for a in pred_indices])
                label_list = " ".join([str(a) for a in label_indices])
                input_list = " ".join([str(a) for a in inputs.tolist()])
                accuracy = 0
                for p, q in zip(pred_indices, label_indices):
                    accuracy = accuracy + (1 if p == q else 0)
                accuracy = accuracy / len(pred_indices) * 100
                average_accuracy += accuracy
                loss = criterion(pred, label)
                validation_log.write(f"{epoch},{step},{input_list},{pred_list},{label_list},{loss.item()},{accuracy}\n")
                wandb.log({
                    "validation/loss": loss.item(),
                    "validation/accuracy": accuracy
                })
        average_accuracy = average_accuracy / len(validation_dataloader)
        wandb.log({
            "accuracy": average_accuracy,
            "epoch": epoch
        })

        checkpoint_dir = os.path.join(save_dir, "latest")
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        backup_dir = os.path.join(save_dir, f"checkpoint-{epoch}")
        for p in [checkpoint_dir, backup_dir]:
            os.makedirs(p, exist_ok=True)
        
        torch.save(model.state_dict(), os.path.join(backup_dir, "model.pth"))
        torch.save(optimizer.state_dict(), os.path.join(backup_dir, "optimizer.pth"))
        torch.save(scheduler.state_dict(), os.path.join(backup_dir, "scheduler.pth"))
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "accuracy": average_accuracy,
            "run_id": wandb.run.id
        }, checkpoint_path)
        wandb.save(checkpoint_path)

            
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    project_name = args.get("project-name", "gene-sequential-labelling")
    run_name = args.get("run-name", "gene-seqlab")
    model_config_dir = args.get("model-config-dir", None)
    model_configs = args.get("model-config", None)
    model_config_names = ", ".join(model_configs)
    device = args.get("device", None)
    device_list = args.get("device-list", [])
    device_names = ", ".join([torch.cuda.get_device_name(a) for a in device_list])
    loss_weight = args.get("loss-weight", None)

    training_config = json.load(open(args.get("training-config", None), "r"))
    batch_size = args.get("batch-size", training_config.get("batch_size", 1))
    num_epochs = args.get("num-epochs", training_config.get("num_epochs", 1))
    training_data = training_config.get("training_data", False)
    training_data = str(pathlib.Path(pathlib.PureWindowsPath(training_data)))
    validation_data = training_config.get("validation_data", False)
    validation_data = str(pathlib.Path(pathlib.PureWindowsPath(validation_data)))
    test_data = training_config.get("test_data", False)
    resume_run_id = args.get("resume-run-id", False)

    for p in [os.path.exists(os.path.join(model_config_dir, f"{a}.csv")) for a in model_configs]:
        if not os.path.exists(p):
            raise ValueError(f"Model configration not found at {p}")
    
    dnabert_path = os.path.join("pretrained", "3-new-12w-0")
    train_dataloader = preprocessing_gene_kmer(training_data, BertTokenizer.from_pretrained(dnabert_path), 1)
    validation_dataloader = preprocessing_gene_kmer(validation_data, BertTokenizer.from_pretrained(dnabert_path), 1)
    n_train_data = len(train_dataloader)
    n_validation_data = len(validation_dataloader)

    print(f"~~~~~Training Gene Sequential Labelling~~~~~")
    print(f"# Training Data {n_train_data}")
    print(f"# Validation Data {n_validation_data}")
    print(f"Device {torch.cuda.get_device_name(device)}")
    print(f"Device List {device_names}")
    print(f"Project Name {project_name}")
    print(f"Model Configs {model_config_names}")
    print(f"Epochs {num_epochs}")

    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    for config in model_configs:
        runname = f"{run_name}-{config}-b{batch_size}-e{num_epochs}-{cur_date}"
        save_dir = os.path.join("run", runname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        model_config = os.path.join(model_config_dir, f"{config}.json")
        model_config = json.load(open(model_config, "r"))
        freeze = model_config.get("freeze_bert", False)
        print(f"Freezing BERT {freeze}")

        bert = BertForMaskedLM.from_pretrained(dnabert_path).bert
        model = DNABERT_RNN(bert, model_config)
        optimizer = AdamW(model.parameters(), 
            lr=training_config["optimizer"]["learning_rate"], 
            betas=(training_config["optimizer"]["beta1"], training_config["optimizer"]["beta1"]),
            eps=training_config["optimizer"]["epsilon"],
            weight_decay=training_config["optimizer"]["weight_decay"]
        )
        training_steps = len(train_dataloader) * num_epochs
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

        run_id = resume_run_id if resume_run_id else wandb.util.generate_id()
        run = wandb.init(project=project_name, entity="anwari32", config={
            "device": device,
            "device_list": device_names,
            "model_config": config,
            "training_data": n_train_data,
            "validation_data": n_validation_data,
            "num_epochs": num_epochs,
            "batch_size": batch_size
        }, reinit=True, resume=True, id=run_id)

        checkpoint_path = os.path.join("run", runname, "latest", "checkpoint.pth")
        start_epoch = 0
        if wandb.run.resumed:
            if os.path.exists(checkpoint_path):
                print(f"Resuming from {run_id}")
                checkpoint = torch.load(wandb.restore(checkpoint_path))
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                start_epoch = checkpoint["epoch"] + 1
                run_id = checkpoint.get("run_id", False)            

        wandb.run.name = f'{runname}-{wandb.run.id}'
        wandb.run.save()
        wandb.watch(model)

        if freeze:
            for param in model.bert.parameters():
                param.requires_grad = False
        
        start_time = datetime.now()
        print(f"Begin Training & Validation {wandb.run.name} at {start_time}")
        train(model, optimizer, scheduler, train_dataloader, validation_dataloader, num_epochs, device, save_dir, wandb, start_epoch, device_list, loss_weight)
        run.finish()
        end_time = datetime.now()
        print(f"Finished Training & Validation at {end_time}")
        print(f"Duration {end_time - start_time}")