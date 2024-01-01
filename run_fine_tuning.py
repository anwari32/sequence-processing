"""
File is named fine-tuning but fine-tuning can be also considered feature-based in case of fine-tuning just last layer. 
Fine-tuning refers to fully update all layers not just last layer.

"""
import sys
import json
from torch.cuda import device_count as cuda_device_count, get_device_name
from torch.optim import AdamW
from sequential_labelling import train
from utils.seqlab import preprocessing
from transformers import BertTokenizer
import os
import wandb
from datetime import datetime
from pathlib import Path, PureWindowsPath
from utils.utils import create_loss_weight
from utils.cli import parse_fine_tuning_command
from torch.optim import lr_scheduler
import torch
from __models.dnabert import DNABertForTokenClassification, num_classes
from utils.seqlab import label2id, id2label
from utils.metrics import Metrics, metric_names
import wandb
from tqdm import tqdm
from torch.cuda.amp import autocast
import pandas as pd
from utils.metrics import clean_prediction_target_batch

def train_and_val(model, 
            optimizer, 
            scheduler, 
            train_dataloader, 
            eval_dataloader,
            num_epoch, 
            save_dir,
            device="cpu", 
            wandb=wandb,
            device_list=[],    
            training_counter=0):

    wandb.define_metric("epoch")
    wandb.define_metric("training_step")
    wandb.define_metric("validation_step")
    wandb.define_metric(f"epoch/*", step_metric="epoch")
    wandb.define_metric(f"training/*", step_metric="training_step")
    wandb.define_metric(f"validation/*", step_metric="validation_step")
    for label_index in range(num_classes):
        label = model.config.id2label[label_index]
        for m in metric_names:
            wandb.define_metric(f"epoch/train-{m}-{label}", step_metric="epoch")
            wandb.define_metric(f"epoch/val-{m}-{label}", step_metric="epoch")
            wandb.define_metric(f"training/{m}-{label}", step_metric="training_step")
            wandb.define_metric(f"validation/{m}-{label}", step_metric="validation_step")
    
    training_step = 0
    validation_step = 0
    tlog_input_ids = []
    tlog_prediction = []
    tlog_target = []
    vlog_input_ids = []
    vlog_prediction = []
    vlog_target = []

    activation = torch.nn.Softmax(2)
    for epoch in range(training_counter, num_epochs):
        epoch_loss = 0
        tt = []
        tp = []
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training [{epoch + 1}/{num_epoch}]"):
            optimizer.zero_grad()
            input_ids, attention_mask, input_type_ids, target_labels = tuple(t.to(device) for t in batch)    
            with autocast():
                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=input_type_ids, labels=target_labels)
                loss = output.loss
                logits = output.logits
                hidden_states = output.hidden_states
                hidden_attentions = output.attentions

            loss.backward()
            epoch_loss += loss
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "training/learning_rate": lr,
                "training/loss": loss.item(),
                "training_step": training_step
                })
            
            x_input_ids = input_ids.view(-1).tolist()
            tlog_input_ids.append(
                " ".join([str(a) for a in x_input_ids])
            )
            y_target = target_labels.view(-1).tolist()
            tlog_target.append(
                " ".join([str(a) for a in y_target])
            )
            y_prediction = activation(logits)
            y_prediction = torch.argmax(y_prediction, 2).view(-1).tolist()
            tlog_prediction.append(
                " ".join([str(a) for a in y_prediction])
            )

            # metrics.
            cy_pred, cy_target = clean_prediction_target_batch(y_prediction, y_target)
            metrics_at_step = Metrics(
                cy_pred, 
                cy_target, 
                num_classes=num_classes)
            metrics_at_step.calculate()
            for label_index in range(num_classes):
                label = model.config.id2label[label_index]
                wandb.log({
                    f"training/precision-{label}": metrics_at_step.precision(label_index),
                    f"training/recall-{label}": metrics_at_step.recall(label_index),
                    f"training/f1_score-{label}": metrics_at_step.f1_score(label_index),
                    "training_step": training_step
                })
            
            # increment training step, accumulate target and prediction.
            training_step += 1
            tp.append(cy_pred)
            tt.append(cy_target)
   
        # move scheduler to epoch loop and compute training metrics for this epoch
        scheduler.step()
        wandb.log({
            "epoch/training_loss": epoch_loss,
            "epoch": epoch
        })
        tp = torch.flatten(torch.tensor(tp)).tolist()
        tt = torch.flatten(torch.tensor(tt)).tolist()
        m = Metrics(tp, tt, num_classes=8)
        m.calculate()
        for label_index in range(model.config.num_labels):
            label_name = model.config.id2label[label_index]
            wandb.log({
                f"epoch/train-precision-{label_name}": m.precision(label_index),
                f"epoch/train-recall-{label_name}": m.recall(label_index),
                f"epoch/train-f1_score-{label_name}": m.f1_score(label_index),
                "epoch": epoch
            })

        
        model.eval()
        vt = []
        vp = []
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Validating"):
                input_ids, attention_mask, token_type_ids, target_labels = (t.to(device) for t in batch)
                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=target_labels)
                loss = output.loss
                logits = output.logits

                x_input_ids = input_ids.view(-1).tolist()
                vlog_input_ids.append(
                    " ".join([str(a) for a in x_input_ids])
                )
                y_target = target_labels.view(-1).tolist()
                vlog_target.append(
                    " ".join([str(a) for a in y_target])
                )
                y_prediction = activation(logits)
                y_prediction = torch.argmax(y_prediction, 2).view(-1).tolist()
                vlog_prediction.append(
                    " ".join([str(a) for a in y_prediction])
                )

                # metrics.
                cy_pred, cy_target = clean_prediction_target_batch(y_prediction, y_target)
                m = Metrics(
                    cy_pred, 
                    cy_target
                )
                m.calculate()
                for label_index in range(num_classes):
                    label = model.config.id2label[label_index]
                    wandb.log({
                        f"validation/precision-{label}": m.precision(label_index),
                        f"validation/recall-{label}": m.recall(label_index),
                        f"validation/f1_score-{label}": m.f1_score(label_index),
                        "validation_step": training_step
                    })
                
                validation_step += 1
                vp.append(cy_pred)
                vt.append(cy_target)

        # save epoch validation.
        vt = torch.flatten(torch.tensor(vt)).tolist()
        vp = torch.flatten(torch.tensor(vp)).tolist()
        m = Metrics(
            vp, 
            vt)
        m.calculate()
        for label_index in range(model.config.num_labels):
            label = model.config.id2label[label_index]
            wandb.log({
                f"epoch/val-precision-{label}": m.precision(label_index),
                f"epoch/val-recall-{label}": m.recall(label_index),
                f"epoch/val-f1_score-{label}": m.f1_score(label_index),
                "epoch": epoch
            })
        
        _model = model.module if isinstance(model, torch.nn.DataParallel) else model
        latest_dir = os.path.join(save_dir, "latest")
        latest_model = os.path.join(latest_dir, "checkpoint.pth")
        if not os.path.exists(latest_dir):
            os.makedirs(latest_dir, exist_ok=True)
        torch.save({
            "model": _model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, latest_model)
        wandb.save(latest_dir)
       
    tframe = pd.DataFrame(data={
        "input_ids": tlog_input_ids,
        "prediction": tlog_prediction,
        "target": tlog_target,
    })
    tframe.to_csv(
        os.path.join(save_dir, "training_log.csv"),
        index=False
    )
    vframe = pd.DataFrame(data={
        "input_ids": vlog_input_ids,
        "prediction": vlog_prediction,
        "target": vlog_target
    })
    vframe.to_csv(
        os.path.join(save_dir, "validation_log.csv"),
        index=False
    )



if __name__ == "__main__":
    print("Command Parameters.")
    args = parse_fine_tuning_command(sys.argv[1:])

    training_config_dirpath = args.get("config-dir", False)
    training_config_names = args.get("config-names", False)
    training_task = args.get("task", False)
    if training_task not in ["seqlab", "seqclass"]:
        raise ValueError(f"Task {training_task} not recognized.")
        
    if not training_config_dirpath or not training_config_names:
        raise ValueError(f"either config path or config names is not detected \n{training_config_dirpath}\n{training_config_names}")
    
    config_paths = []
    for name in training_config_names:
        config_paths.append(os.path.join(training_config_dirpath, f"{name}.json"))

    if not all([os.path.exists(p) for p in config_paths]):
        raise FileNotFoundError(f"not all config paths exist")

    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")

    num_epochs = 0
    batch_size = 0
    train_filepath = None
    validation_filepath = None
    train_dataloader = None
    validation_dataloader = None
    for c in config_paths:
        tconfig = json.load(open(c, "r"))

        # Override batch size and epoch if given in command.
        num_epochs = tconfig.get("num_epochs") if not tconfig.get("num_epochs") == num_epochs else num_epochs
        batch_size = tconfig.get("batch_size") if not tconfig.get("batch_size") == batch_size else batch_size

        _train_filepath = str(Path(PureWindowsPath(tconfig.get("train_data", False))))
        if not _train_filepath == train_filepath:
            train_filepath = _train_filepath
            pretrained_path = os.path.join("pretrained", "3-new-12w-0")
            tokenizer = BertTokenizer.from_pretrained(pretrained_path)
            train_dataloader = preprocessing(
                train_filepath,# csv_file, 
                tokenizer,
                batch_size,
                do_kmer=False
                )
            n_train_data = len(train_dataloader)

        _vpath = str(Path(PureWindowsPath(tconfig.get("validation_data", False))))
        if not _vpath == validation_filepath:
            validation_filepath = _vpath
            eval_dataloader = preprocessing(
                validation_filepath,# csv_file, 
                tokenizer,
                batch_size,
                do_kmer=False
            )
            n_validation_data = len(eval_dataloader)

    # All training devices are CUDA GPUs.
    device = args.get("device", "cpu")
    device_name = get_device_name(device)
    device_names = ""
    device_list = [int(a) for a in args.get("device-list", [])]
    if "device-list" in args.keys():
        print(f"# GPU: {len(device_list)}")
        device_names = ", ".join([get_device_name(f"cuda:{a}") for a in device_list])

    # Run name may be the same. So append current datetime to differentiate.
    # Create this folder if not exist.
    project_name = args.get("project-name", "DNABertForTokenClassification")
    use_weighted_loss = args.get("use-weighted-loss", False)
    loss_weight = create_loss_weight(train_filepath) if use_weighted_loss else None
    
    run_name = args.get("run-name", tconfig.get("name", "dnabertfortokenclassification"))    
    model = DNABertForTokenClassification.from_pretrained(
        pretrained_path,
        num_labels = num_classes,
        id2label = id2label,
        label2id = label2id)
    model.to(device)

    # enable data parallel if possible
    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)

    optimizer_config = tconfig.get("optimizer", None)
    learning_rate = args.get("lr", optimizer_config.get("lr", 1e-3))
    optimizer = None
    if not optimizer_config:
        # set to default implementation.
        optimizer = AdamW(model.parameters())
    else:
        optimizer = AdamW(model.parameters(), 
            lr=learning_rate,
            betas=(
                optimizer_config.get("beta1", 0.9), 
                optimizer_config.get("beta2", 0.98)
            ),
            eps=optimizer_config.get("epsilon", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 0.01)
        )

    training_steps = len(train_dataloader) * num_epochs
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # Prepare wandb.
    os.environ["WANDB_MODE"] = "offline" if args.get("offline", False) else "online"
    run_id = wandb.util.generate_id()
    runname = f"{run_name}-{run_id}"
    run = wandb.init(project=project_name, entity="anwari32", config={
        "device": device,
        "device_list": device_names,
        "model_config": "DNABertForTokenClassification",
        "training_data": n_train_data,
        "validation_data": n_validation_data,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "training_date": cur_date,
        "initial_learning_rate": learning_rate,
        "use_weighted_loss": use_weighted_loss,
    }, reinit=True, resume='allow', id=run_id, name=runname) 
    wandb.watch(model)

    save_dir = os.path.join("run", runname)
    checkpoint_dir = os.path.join(save_dir, "latest")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    print(f"Save Directory {save_dir}")
    print(f"Checkpoint Directory {checkpoint_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    if run.resumed:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1

    start_time = datetime.now()
    print(f"~~~~~Training Sequential Labelling~~~~~")
    print(f"# Training Data {n_train_data}")
    print(f"# Validation Data {n_validation_data}")
    print(f"Device {torch.cuda.get_device_name(device)}")
    print(f"Device List {device_names}")
    print(f"Project Name {project_name}")
    print(f"Epochs {num_epochs}")
    print(f"Use weighted loss {use_weighted_loss}")
    print(f"Initial Learning Rate {learning_rate}")
    print(f"Begin Training & Validation {wandb.run.name} at {start_time}")
    print(f"Starting epoch {start_epoch}")

    train_and_val(
        model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        eval_dataloader,
        num_epochs, 
        save_dir,
        device=device, 
        wandb=wandb,
        device_list=[],    
        training_counter=0
    )
    run.finish()
    end_time = datetime.now()
    running_time = end_time - start_time
    print(f"Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")
