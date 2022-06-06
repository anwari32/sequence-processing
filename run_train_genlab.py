from getopt import getopt
import json
from sched import scheduler
import sys
import os
from transformers import BertForMaskedLM
from sequential_gene_labeling import train
from utils.model import init_seqlab_model
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda import device_count as cuda_device_count
from utils.tokenizer import get_default_tokenizer
import wandb
import pandas as pd
from pathlib import Path, PureWindowsPath
from datetime import datetime

from utils.utils import save_checkpoint

def _parse_argv(argvs):
    opts, args = getopt(argvs, "t:m:d:f:r", [
        "training-config=", 
        "model-config=", 
        "device=", 
        "force-cpu", 
        "resume=",
        "run-name=",
        "device-list=",
        "disable-wandb",
        "num-epochs=",
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
        elif o in ["r", "--resume"]:
            output["resume"] = a
        elif o in ["--device-list"]:
            output["device_list"] = [int(x) for x in a.split(",")]
        elif o in ["--run-name"]:
            output["run_name"] = a
        elif o in ["--disable-wandb"]:
            output["disable_wandb"] = True
        elif o in ["--num-epochs"]:
            output["num_epochs"] = int(a)
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
    assert "run_name" in args.keys(), f"run_name must be stated."
    
    # Run name may be the same. So append current datetime to differentiate.
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    args["run_name"] = f"{args['run_name']}-{cur_date}"

    training_config_path = str(Path(PureWindowsPath(args["training_config"])))
    training_config = json.load(open(training_config_path, "r"))
    
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

    # TODO: develop resume training feature here.
    load_from_dirpath = os.path.join(args["resume"])
    
    # Simplify optimizer, just use default parameters if necessary.
    lr = training_config["optimizer"]["learning_rate"]
    optimizer = AdamW(model.parameters(), 
        lr=training_config["optimizer"]["learning_rate"], 
        betas=(training_config["optimizer"]["beta1"], training_config["optimizer"]["beta1"]),
        eps=training_config["optimizer"]["epsilon"],
        weight_decay=training_config["optimizer"]["weight_decay"]
    )

    # Define loss function.
    loss_function = torch.nn.CrossEntropyLoss()

    # Define training and evaluation data.
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
    print(f"# Training Genes: {len(train_genes)}")
    print(f"# Validation Genes: {len(eval_genes)}")
    training_steps = len(train_genes) * training_config["num_epochs"]

    # Use scheduler from torch implementation for the sake of simplicity.
    scheduler = ExponentialLR(optimizer, gamma=0.1)

    # Freeze BERT layer if necessary.    
    if "freeze_bert" in training_config.keys():
        if training_config["freeze_bert"] > 0:
            print(">> Freeze BERT layer.", end="\r")
            for param in model.bert.parameters():
                param.requires_grad = False
            print(f">> Freeze BERT layer. [{all([p.requires_grad == False for p in model.bert.parameters()])}]")

    if "device_list" in args.keys():
        print(f"# GPU: {len(args['device_list'])}")
    
    batch_size = training_config["batch_size"] if "batch_size" not in args.keys() else args["batch_size"]
    num_epochs = training_config["num_epochs"] if "num_epochs" not in args.keys() else args["num_epochs"]

    # Prepare wandb.
    args["disable_wandb"] = True if "disable_wandb" in args.keys() else False
    os.environ["WANDB_MODE"] = "offline" if args["disable_wandb"] else "online"
    wandb.init(project="thesis-mtl", entity="anwari32", config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size
    }) 
    if "run_name" in args.keys():
        wandb.run.name = f'{args["run_name"]}-{wandb.run.id}'
        wandb.run.save()
    wandb.watch(model)

    save_dir = os.path.join("run", args["run_name"])
    os.makedirs(save_dir, exist_ok=True)
    
    # Save current model config in run folder.
    model_config = json.load(open(str(Path(PureWindowsPath(args["model_config"]))), "r"))
    json.dump(model_config, open(os.path.join("run", args["run_name"], "model_config.json"), "x"), indent=4)

    start_time = datetime.now()
    end_time = start_time
    try:
        model, optimizer = train(
            model=model, 
            tokenizer=get_default_tokenizer(),
            optimizer=optimizer, 
            scheduler=scheduler, 
            train_genes=train_genes, 
            loss_function=loss_function, 
            num_epoch=num_epochs, 
            batch_size=batch_size, 
            device=args["device"],
            wandb=wandb,
            save_dir=save_dir,
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

    # Save final model and optimizer.
    save_checkpoint(model, optimizer, scheduler, total_config, os.path.join(save_dir, "final-checkpoint.pth"))
    
    

