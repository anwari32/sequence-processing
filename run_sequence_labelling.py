from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count, get_device_name
from torch.optim import AdamW
from sequential_labelling import train
from utils.seqlab import preprocessing
from utils.model import init_seqlab_model
from transformers import BertTokenizer
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
    for key in args.keys():
        print(f"- {key} {args[key]}")

    # Make sure input config parameters are valid.
    if "training_config" not in args.keys():
        print("Please provide training config.")
        sys.exit(2)

    if not os.path.exists(args["training_config"]):
        print(f"Training config not found at {args['training_config']}")
        sys.exit(2)
    
    if "model_config" not in args.keys() and "model_config_dir" not in args.keys() and "model_config_names" not in args.keys():
        print(f"Model config not found at {args['model_config']}")
        sys.exit(2)

    # Make sure input parameters are valid.
    if not "force-cpu" in args.keys():
        if args["device"] == "cpu":
            print(f"Don't use CPU for training")
            sys.exit(2)
        cuda_device_count = cuda_device_count()
        if cuda_device_count > 1 and args["device"] == "cuda":
            print(f"There are more than one CUDA devices. Please choose one.")
            sys.exit(2)
    
    # Run name is made required. If there is None then Error shall be there.
    if not "run_name" in args.keys():
        print("Run name is required.")
        print("`--run-name=<runname>`")
        sys.exit(2)

    training_config_path = args["training_config"]
    training_config = json.load(open(training_config_path, "r"))

    # Override batch size and epoch if given in command.
    epoch_size = training_config["num_epochs"] if "num_epochs" not in args.keys() else args["num_epochs"]
    batch_size = training_config["batch_size"] if "batch_size" not in args.keys() else args["batch_size"]

    training_filepath = str(Path(PureWindowsPath(training_config["train_data"])))
    validation_filepath = str(Path(PureWindowsPath(training_config["validation_data"])))
    print(f"Preparing Training Data {training_filepath}")
    dataloader = preprocessing(
        training_filepath,# csv_file, 
        BertTokenizer.from_pretrained(str(Path(PureWindowsPath(training_config["pretrained"])))), # tokenizer
        batch_size, # training_config["batch_size"], #batch_size,
        do_kmer=False
        )
    n_train_data = len(dataloader)
    print(f"Preparing Validation Data {validation_filepath}")
    eval_dataloader = preprocessing(
        validation_filepath,# csv_file, 
        BertTokenizer.from_pretrained(str(Path(PureWindowsPath(training_config["pretrained"])))), # tokenizer
        batch_size, #1,
        do_kmer=False
    )
    n_validation_data = len(eval_dataloader)

    # All training devices are CUDA GPUs.
    device = args["device"]
    device_name = get_device_name(device)
    device_names = ""
    device_list = []
    if "device_list" in args.keys():
        print(f"# GPU: {len(args['device_list'])}")
        device_list = args["device_list"]
        device_names = ", ".join([get_device_name(f"cuda:{a}") for a in device_list])

    # Run name may be the same. So append current datetime to differentiate.
    # Create this folder if not exist.
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_config_list = []
    if "model_config_dir" in args.keys() and "model_config_names" in args.keys():
        model_config_list = [os.path.join(args["model_config_dir"], f"{n}.json") for n in args["model_config_names"]]
        if not all([os.path.exists(p) for p in model_config_list]):
            raise FileNotFoundError("Path to model config not found")
    else:
        if not os.path.exists(args["model_config"]):
            raise FileNotFoundError("Path to model config not found")
        model_config_list.append(args["model_config"])

    args["disable_wandb"] = True if "disable_wandb" in args.keys() else False
    os.environ["WANDB_MODE"] = "offline" if args["disable_wandb"] else "online"
    project_name = args.get("project_name", "thesis")    
    use_weighted_loss = args.get("use-weighted-loss", False)
    loss_weight = create_loss_weight(training_filepath) if use_weighted_loss else None
    resume_run_ids = args.get("resume-run-id", [None for a in model_config_list])
    model_config_names = ", ".join(args["model_config_names"])

    print(f"~~~~~Training Sequential Labelling~~~~~")
    print(f"# Training Data {n_train_data}")
    print(f"# Validation Data {n_validation_data}")
    print(f"Device {torch.cuda.get_device_name(device)}")
    print(f"Device List {device_names}")
    print(f"Project Name {project_name}")
    print(f"Model Configs {model_config_names}")
    print(f"Epochs {epoch_size}")
    print(f"Use weighted loss {use_weighted_loss}")

    for cfg_path, resume_run_id in zip(model_config_list, resume_run_ids):
        cfg_name = os.path.basename(cfg_path).split(".")[0:-1] # Get filename without extension.
        if isinstance(cfg_name, list):
            cfg_name = ".".join(cfg_name)
        print(f"Training model with config {cfg_name}")
        run_name = args.get("run_name")
        
        lr = training_config["optimizer"]["learning_rate"]
        model = init_seqlab_model(cfg_path)
        optimizer = AdamW(model.parameters(), 
            lr=training_config["optimizer"]["learning_rate"], 
            betas=(training_config["optimizer"]["beta1"], training_config["optimizer"]["beta1"]),
            eps=training_config["optimizer"]["epsilon"],
            weight_decay=training_config["optimizer"]["weight_decay"]
        )

        training_steps = len(dataloader) * epoch_size
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

        # Prepare wandb.
        run_id = resume_run_id if resume_run_id else wandb.util.generate_id()
        run = wandb.init(project=project_name, entity="anwari32", config={
            "device": device,
            "device_list": device_names,
            "model_config": cfg_name,
            "training_data": n_train_data,
            "validation_data": n_validation_data,
            "num_epochs": epoch_size,
            "batch_size": batch_size,
            "training_date": cur_date
        }, reinit=True, resume='allow', id=run_id) 
        
        runname = f"{run_name}-{cfg_name}-{run_id}"
        save_dir = os.path.join("run", runname)
        print(f"Save Directory {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join("run", runname, "latest", "checkpoint.pth")
        start_epoch = 0
        if wandb.run.resumed:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model"])
                model.to(device)
                optimizer.load_state_dict(checkpoint["optimizer"])
                optimizer.load_state_dict(optimizer.state_dict())
                scheduler.load_state_dict(checkpoint["scheduler"])
                scheduler.load_state_dict(scheduler.state_dict())
                start_epoch = int(checkpoint["epoch"]) + 1

        wandb.run.name = runname
        wandb.run.save()
        wandb.watch(model)

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
            loss_strategy="sum",
            wandb=wandb,
            device_list=device_list,
            eval_dataloader=eval_dataloader,    
            training_counter=start_epoch,
        )
        run.finish()
        end_time = datetime.now()
        running_time = end_time - start_time
        print(f"Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")
