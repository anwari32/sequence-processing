from ast import arg
from genericpath import exists
from getopt import getopt
import json
import sys
import os
from transformers import BertForMaskedLM, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from sequential_labelling import train_by_genes
from utils.model import init_seqlab_model
from utils.optimizer import init_optimizer
import torch
from torch.cuda import device_count as cuda_device_count
from utils.tokenizer import get_default_tokenizer
from utils.utils import load_checkpoint, load_mtl_model, save_checkpoint, save_json_config
import wandb
import pandas as pd
from pathlib import Path, PureWindowsPath
from datetime import datetime

def _parse_argv(argvs):
    opts, args = getopt(argvs, "t:m:d:f", [
        "training-config=", 
        "model-config=", 
        "device=", 
        "force-cpu", 
        "training-counter=", 
        "resume-from-checkpoint=", 
        "resume-from-optimizer=", 
        "run-name=",
        "device-list=",
        "disable-wandb",
    ])
    output = {}
    for o, a in opts:
        if o in ["-t", "--training-config"]:
            output["training_config"] = a
        elif o in ["-m", "--model-config"]:
            output["model_config"] = a
        elif o in ["-d", "--device"]:
            output["device"] = a
        elif o in ["-f", "--force-cpu"]:
            output["force-cpu"] = True
        elif o in ["--training-counter"]:
            output["training_counter"] = a
        elif o in ["--resume-from-checkpoint"]:
            output["resume_from_checkpoint"] = a
        elif o in ["--device-list"]:
            output["device_list"] = [int(x) for x in a.split(",")]
        elif o in ["--run-name"]:
            output["run_name"] = a
        elif o in ["--disable-wandb"]:
            output["disable_wandb"] = True
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output


#   TODO:
#   Implements `learning_rate`, `beta1`, `beta2`, and `weight_decay` on AdamW optimizer.
#   Implements `loss_function`.
#   Implements  `scheduler`
if __name__ == "__main__":
    print("Training Sequential Labelling model with Genes.")
    args = _parse_argv(sys.argv[1:])
    for key in args.keys():
        print(key, args[key])

    # Make sure input parameters are valid.
    if not os.path.exists(args["training_config"]) or not os.path.isfile(args["model_config"]):
        print(f"Training config not found at {args['training_config']}")
    
    if not os.path.exists(args["model_config"]) or not os.path.isfile(args["model_config"]):
        print(f"Model config not found at {args['model_config']}")

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
    
    # Run name may be the same. So append current datetime to differentiate.
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    args["run_name"] = f"{args['run_name']}-{cur_date}"

    training_config_path = str(Path(PureWindowsPath(args["training_config"])))
    training_config = json.load(open(training_config_path, "r"))
    # if training_config["result"] == "":
    #    print(f"Key `result` not found in config.")
    #    sys.exit(2)
    
    result_path = os.path.join("run", args["run_name"])
    # result_path = str(Path(PureWindowsPath(training_config["result"])))
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    print("Initializing DNABERT-SL")
    model = init_seqlab_model(args["model_config"])
    if not "mtl" in training_config.keys():
        print(">> Initializing default DNABERT-SL.")
    else:
        if not training_config["mtl"] == "":
            print(f">> Initializing DNABERT-SL with MTL {training_config['mtl']}")
            # Load BERT layer from MTL-trained folder.
            formatted_path = str(Path(PureWindowsPath(training_config["mtl"])))
            saved_model = BertForMaskedLM.from_pretrained(formatted_path)
            model.bert = saved_model.bert
        else:
            print(">> Invalid DNABERT-MTL result path. Initializing default DNABERT-SEL.")

    if "freeze_bert" in training_config.keys():
        if training_config["freeze_bert"] > 0:
            print(">> Freeze BERT layer.", end="\r")
            for param in model.bert.parameters():
                param.requires_grad = False
            print(f">> Freeze BERT layer. [{all([p.requires_grad == False for p in model.bert.parameters()])}]")

    model.to(args["device"])
    optimizer = init_optimizer(
        training_config["optimizer"]["name"], 
        model.parameters(), 
        training_config["optimizer"]["learning_rate"], 
        training_config["optimizer"]["epsilon"], 
        training_config["optimizer"]["beta1"], 
        training_config["optimizer"]["beta2"], 
        training_config["optimizer"]["weight_decay"]
    )

    training_counter = args["training_counter"] if "training_counter" in args.keys() else 0
    if "resume_checkpoint" in args.keys():
        checkpoint = load_checkpoint(args["resume_from_checkpoint"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        training_counter = int(checkpoint["epoch"]) + 1

    loss_function = torch.nn.CrossEntropyLoss()

    train_df = pd.read_csv(str(Path(PureWindowsPath(training_config["gene_train_index"]))))
    validation_df = pd.read_csv(str(Path(PureWindowsPath(training_config["gene_validation_index"]))))
    train_genes = []
    for i, r in train_df.iterrows():
        train_genes.append(
            str(Path(PureWindowsPath(os.path.join(training_config["gene_dir"], r["chr"], r["gene"]))))
        )    

    eval_genes = []
    for i, r in validation_df.iterrows():
        eval_genes.append(
            str(Path(PureWindowsPath(os.path.join(training_config["gene_dir"], r["chr"], r["gene"]))))
        )

    print(f"# Genes: {len(train_genes)}")
    training_steps = len(train_genes) * training_config["num_epochs"]
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)

    if "device_list" in args.keys():
        print(f"# GPU: {len(args['device_list'])}")
    
    if "disable_wandb" not in args.keys():
        args["disable_wandb"] = False
    
    if args["disable_wandb"]:
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ["WANDB_MODE"] = "online"

    wandb.init(project="thesis-mtl", entity="anwari32") 
    if "run_name" in args.keys():
        wandb.run.name = f'{args["run_name"]}-{wandb.run.id}'
        wandb.run.save()
    wandb.config = {
        "learning_rate": training_config["optimizer"]["learning_rate"],
        "epochs": training_config["num_epochs"],
        "batch_size": training_config["batch_size"]
    }
        
    # log_dir_path = str(Path(PureWindowsPath(training_config["log"])))
    # log_file_path = os.path.join(log_dir_path, "by_genes", cur_date, "log.csv")
    log_file_path = os.path.join("run", args["run_name"], "logs", "log.csv")

    # save_model_path = str(Path(PureWindowsPath(training_config["result"])))
    # save_model_path = os.path.join(save_model_path, "by_genes", cur_date)
    save_model_path = os.path.join("run", args["run_name"])

    for p in [log_file_path, save_model_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    
    # Save current model config in run folder.
    model_config = json.load(open(str(Path(PureWindowsPath(args["model_config"]))), "r"))
    json.dump(model_config, open(os.path.join("run", args["run_name"], "model_config.json"), "x"), indent=4)

    start_time = datetime.now()
    end_time = start_time
    try:
        model, optimizer = train_by_genes(
            model=model, 
            tokenizer=get_default_tokenizer(),
            optimizer=optimizer, 
            scheduler=scheduler, 
            train_genes=train_genes, 
            loss_function=loss_function, 
            num_epoch=training_config["num_epochs"], 
            batch_size=training_config["batch_size"], 
            grad_accumulation_steps=training_config["grad_accumulation_steps"],
            device=args["device"],
            wandb=wandb,
            save_path=save_model_path,
            log_file_path=log_file_path,
            training_counter=training_counter,
            eval_genes=eval_genes,
            device_list=args["device_list"] if "device_list" in args.keys() else [])
    except Exception as ex:
        print(ex)
        end_time = datetime.now()
        running_time = end_time - start_time
        print(f"Error: Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")    

    end_time = datetime.now()
    running_time = end_time - start_time

    print(f"Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")

    total_config = {
        "training": training_config,
        "model": json.load(open(str(Path(PureWindowsPath(args["model_config"]))), "r")),
        "start_time": start_time.strftime("%Y%m%d-%H%M%S"),
        "end_time": end_time.strftime("%Y%m%d-%H%M%S"),
        "running_time": str(running_time),
        "runname": args["run_name"]
    }

    # Final config is saved in JSON format in the same folder as log file.
    # Final config is saved in `config.json` file.
    # save_json_config(total_config, os.path.join(os.path.dirname(str(Path(PureWindowsPath(training_config["log"])))), "config.json"))
    save_json_config(total_config, os.path.join("run", args["run_name"], "config.json"))

    # Save final model and optimizer.
    save_checkpoint(model, optimizer, total_config, os.path.join(save_model_path, "final-checkpoint.pth"))
    
    

