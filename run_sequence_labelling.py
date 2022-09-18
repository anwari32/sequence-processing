from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count, get_device_name
from torch.optim import AdamW
from models.seqlab import DNABERT_SL
from sequential_labelling import train
from utils.seqlab import preprocessing
from transformers import BertTokenizer, BertForMaskedLM
import os
import wandb
from datetime import datetime
from pathlib import Path, PureWindowsPath
from utils.utils import create_loss_weight, save_checkpoint
from utils.cli import parse_args
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
import torch
import traceback

if __name__ == "__main__":
    print("Command Parameters.")
    args = parse_args(sys.argv[1:])

    training_config_path = args.get("training-config", False)
    training_config = json.load(open(training_config_path, "r"))

    # Override batch size and epoch if given in command.
    epoch_size = args.get("num-epochs", training_config.get("num_epochs", 1))
    batch_size = args.get("batch-size", training_config.get("batch_size", 1))

    training_filepath = str(Path(PureWindowsPath(training_config.get("train_data", False))))
    validation_filepath = str(Path(PureWindowsPath(training_config.get("validation_data", False))))
    pretrained = str(Path(PureWindowsPath(training_config["pretrained"]))) if "pretrained" in training_config.keys() else os.path.join("pretrained", "3-new-12w-0")
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    dataloader = preprocessing(
        training_filepath,# csv_file, 
        tokenizer, # tokenizer
        batch_size, # training_config["batch_size"], #batch_size,
        do_kmer=False
        )
    n_train_data = len(dataloader)
    eval_dataloader = preprocessing(
        validation_filepath,# csv_file, 
        tokenizer, # tokenizer
        batch_size, #1,
        do_kmer=False
    )
    n_validation_data = len(eval_dataloader)

    # All training devices are CUDA GPUs.
    device = args.get("device", False)
    device_name = get_device_name(device)
    device_names = ""
    device_list = []
    if "device-list" in args.keys():
        print(f"# GPU: {len(args.get('device-list'))}")
        device_list = args.get("device-list")
        device_names = ", ".join([get_device_name(f"cuda:{a}") for a in device_list])

    # Run name may be the same. So append current datetime to differentiate.
    # Create this folder if not exist.
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_config_names = args.get("model-config-names", False)
    model_config_dir = args.get("model-config-dir", False)
    model_config_list = []
    if "model-config-dir" in args.keys() and "model-config-names" in args.keys():
        model_config_list = [os.path.join(model_config_dir, f"{n}.json") for n in model_config_names]
        if not all([os.path.exists(p) for p in model_config_list]):
            raise FileNotFoundError("Path to model config not found")
    else:
        if not os.path.exists(args.get("model-config", False)):
            raise FileNotFoundError("Path to model config not found")
        model_config_list.append(args.get("model-config", False))

    project_name = args.get("project-name", "sequence-labelling")    
    use_weighted_loss = args.get("use-weighted-loss", False)
    loss_weight = create_loss_weight(training_filepath) if use_weighted_loss else None
    resume_run_ids = args.get("resume-run-id", [])
    
    n_config_names = len(model_config_names)
    n_resume_run_ids = len(resume_run_ids)
    if (n_resume_run_ids < n_config_names):
        delta = n_config_names - n_resume_run_ids
        for i in range(delta):
            resume_run_ids.append(None)

    model_config_names = ", ".join(model_config_names)
    run_name = args.get("run-name", "sequence-labelling")
    learning_rate = args.get("lr", None)

    print(f"~~~~~Training Sequential Labelling~~~~~")
    print(f"# Training Data {n_train_data}")
    print(f"# Validation Data {n_validation_data}")
    print(f"Device {torch.cuda.get_device_name(device)}")
    print(f"Device List {device_names}")
    print(f"Project Name {project_name}")
    print(f"Model Configs {model_config_names}")
    print(f"Epochs {epoch_size}")
    print(f"Use weighted loss {use_weighted_loss}")
    print(f"Initial Learning Rate {learning_rate}")

    for cfg_path, resume_run_id in zip(model_config_list, resume_run_ids):
        cfg_name = os.path.basename(cfg_path).split(".")[0:-1] # Get filename without extension.
        if isinstance(cfg_name, list):
            cfg_name = ".".join(cfg_name)
        print(f"Training model with config {cfg_name}")
        
        pretrained = os.path.join("pretrained", "3-new-12w-0")
        if "pretrained" in training_config.keys():
            pretrained = str(Path(PureWindowsPath(training_config.get("pretrained", False))))
        
        _config = json.load(open(cfg_path, "r"))
        _bert = BertForMaskedLM.from_pretrained(pretrained)
        _bert = _bert.bert
        model = DNABERT_SL(_bert, _config)
        if learning_rate == None:
            learning_rate = training_config["optimizer"]["learning_rate"]

        optimizer = AdamW(model.parameters(), 
            lr=learning_rate, 
            betas=(
                training_config["optimizer"]["beta1"], 
                training_config["optimizer"]["beta1"]
            ),
            eps=training_config["optimizer"]["epsilon"],
            weight_decay=training_config["optimizer"]["weight_decay"]
        )

        training_steps = len(dataloader) * epoch_size
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

        # Prepare wandb.
        os.environ["WANDB_MODE"] = "offline" if args.get("offline", False) else "online"
        run_id = resume_run_id if resume_run_id != None else wandb.util.generate_id()
        runname = f"{run_name}-{cfg_name}-{run_id}"
        run = wandb.init(project=project_name, entity="anwari32", config={
            "device": device,
            "device_list": device_names,
            "model_config": cfg_name,
            "training_data": n_train_data,
            "validation_data": n_validation_data,
            "num_epochs": epoch_size,
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

        # Save current model config in run folder.
        model_config = json.load(open(cfg_path, "r"))
        model_config_path = os.path.join(save_dir, "model_config.json")
        json.dump(model_config, open(model_config_path, "x"), indent=4)

        # Save current training config in run folder.
        training_config_path = os.path.join(save_dir, "training_config.json")
        json.dump(training_config, open(training_config_path, "x"), indent=4)
    
        # Loss function.
        if loss_weight != None:
            loss_weight = loss_weight.to(device)
        loss_function = CrossEntropyLoss(weight=loss_weight)

        if "freeze_bert" in model_config.keys():
            if int(model_config["freeze_bert"]):
                print("Freezing BERT")
                for param in model.bert.parameters():
                    param.requires_grad = False

        start_time = datetime.now()
        print(f"Begin Training & Validation {wandb.run.name} at {start_time}")
        print(f"Starting epoch {start_epoch}")
        trained_model, trained_optimizer, trained_scheduler = train(
            model, 
            optimizer, 
            scheduler, 
            dataloader, 
            epoch_size, 
            save_dir,
            loss_function,
            device=device, 
            wandb=wandb,
            device_list=device_list,
            eval_dataloader=eval_dataloader,    
            training_counter=start_epoch,
        )
        run.finish()
        end_time = datetime.now()
        running_time = end_time - start_time
        print(f"Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")
