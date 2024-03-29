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
from __models.genlab import DNABERT_RNN
from utils.utils import create_loss_weight
from utils.cli import parse_args

def train(model, optimizer, scheduler, train_dataloader, validation_dataloader, num_epochs, device, save_dir, wandb, start_epoch=0, device_list=[], criterion_weight=None):
    model.to(device)
    if criterion_weight != None:
        criterion_weight = criterion_weight.to(device)
        # print("Criterion location ", criterion_weight.device)
    num_labels = model.num_labels
    criterion = CrossEntropyLoss(weight=criterion_weight)
    
    n_train_data = len(train_dataloader)
    n_validation_data = len(validation_dataloader)
    training_log_path = os.path.join(save_dir, "training_log.csv")
    if os.path.exists(training_log_path):
        os.remove(training_log_path)
    training_log = open(training_log_path, "x")
    training_log.write("epoch,step,sequence,prediction,target,loss\n")

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
    
        validation_log_path = os.path.join(save_dir, f"validation_log.{epoch}.csv")
        if os.path.exists(validation_log_path):
            os.remove(validation_log_path)
        validation_log = open(validation_log_path, "x")
        validation_log.write("epoch,step,sequence,prediction,target,loss,accuracy\n")
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
        
        # EDIT 21 August 2022: Remove saving model, optimizer, and scheduler for each epoch to save disk space.
        #torch.save(model.state_dict(), os.path.join(backup_dir, "model.pth"))
        #torch.save(optimizer.state_dict(), os.path.join(backup_dir, "optimizer.pth"))
        #torch.save(scheduler.state_dict(), os.path.join(backup_dir, "scheduler.pth"))
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
    model_configs = args.get("model-config-names", None)
    model_config_names = ", ".join(model_configs)
    device = args.get("device", None)
    device_list = args.get("device-list", [])
    device_names = ", ".join([torch.cuda.get_device_name(a) for a in device_list])
    use_weighted_loss = args.get("use-weighted-loss", False)
    resume_run_ids = args.get("resume-run-ids", [])
    training_config = json.load(open(args.get("training-config", None), "r"))
    batch_size = args.get("batch-size", training_config.get("batch_size", 1))
    num_epochs = args.get("num-epochs", training_config.get("num_epochs", 1))
    training_data = training_config.get("training_data", False)
    training_data = str(pathlib.Path(pathlib.PureWindowsPath(training_data)))
    validation_data = training_config.get("validation_data", False)
    validation_data = str(pathlib.Path(pathlib.PureWindowsPath(validation_data)))
    test_data = training_config.get("test_data", False)
    loss_weight = create_loss_weight(training_data) if use_weighted_loss else None
    
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
    print(f"Use weighted loss {use_weighted_loss}")

    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    if resume_run_ids == []:
        resume_run_ids = [None for m in model_configs]

    for config, resume_run_id in zip(model_configs, resume_run_ids):
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
            "batch_size": batch_size,
            "training_date": cur_date
        }, reinit=True, resume='allow', id=run_id)

        runname = f"{run_name}-{config}-{run_id}"
        save_dir = os.path.join("run", runname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join("run", runname, "latest", "checkpoint.pth")
        start_epoch = 0
        if wandb.run.resumed:
            if os.path.exists(checkpoint_path):
                print(f"Resuming from {run_id}")
                # checkpoint = torch.load(wandb.restore(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                start_epoch = checkpoint["epoch"] + 1

        wandb.run.name = f'{runname}'
        wandb.run.save()
        wandb.watch(model)

        if freeze:
            for param in model.bert.parameters():
                param.requires_grad = False
        
        start_time = datetime.now()
        print(f"Begin Training & Validation {wandb.run.name} at {start_time}")
        print(f"Starting epoch {start_epoch}")
        train(model, optimizer, scheduler, train_dataloader, validation_dataloader, num_epochs, device, save_dir, wandb, start_epoch, device_list, loss_weight)
        run.finish()
        end_time = datetime.now()
        print(f"Finished Training & Validation at {end_time}")
        running_time = end_time - start_time
        print(f"Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")