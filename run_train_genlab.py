from getopt import getopt
import json
from sched import scheduler
import sys
import os
from transformers import BertForMaskedLM
from models.genlab import DNABERT_GSL
from sequential_gene_labelling import train
from utils.model import init_seqlab_model
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda import device_count as cuda_device_count
from utils.tokenizer import get_default_tokenizer
import wandb
import pandas as pd
from pathlib import Path, PurePath, PureWindowsPath
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
        "batch-size=",
        "model-config-dir=",
        "model-config-names=",
        "project-name="
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
        elif o in ["--batch-size"]:
            output["batch_size"] = int(a)
        elif o in ["--model-config-dir"]:
            output["model_config_dir"] = a
        elif o in ["--model-config-names"]:
            output["model_config_names"] = a.split(',')
        elif o in ["--project-name"]:
            output["project_name"] = a
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

if __name__ == "__main__":
    print("Training Sequential Labelling model with Genes.")
    args = _parse_argv(sys.argv[1:])
    for key in args.keys():
        print(f"- {key} {args[key]}")

    # Make sure input parameters are valid.
    assert os.path.exists(args["training_config"]) and os.path.isfile(args["training_config"]), f"Training config not found at {args['training_config']}"
    #assert os.path.exists(args["model_config"]) and os.path.isfile(args["model_config"]), f"Model config not found at {args['model_config']}"

    # Run name is made required. If there is None then Error shall be there.
    assert "run_name" in args.keys(), f"run_name must be stated."

    if not "force-cpu" in args.keys():
        if args["device"] == "cpu":
            print(f"Don't use CPU for training")
            sys.exit(2)
        cuda_device_count = cuda_device_count()
        if cuda_device_count > 1 and args["device"] == "cuda":
            print(f"There are more than one CUDA devices. Please choose one.")
            sys.exit(2)

    device_list = []
    device_names = ""
    if "device_list" in args.keys():
        print(f"# GPU: {len(args['device_list'])}")
        device_list = args["device_list"]
        device_names = ", ".join([get_device_name(f"cuda:{a}") for a in device_list])

    # All training devices are CUDA GPUs.
    device_name = ""
    if "device" in args.keys():
        device_name = torch.cuda.get_device_name(args["device"])

    device_names = ""
    if "device_list" in args.keys():
        print(f"# GPU: {len(args['device_list'])}")
        device_names = ", ".join([torch.cuda.get_device_name(f"cuda:{a}") for a in args["device_list"]])

    
    # Determine batch size and epochs.    
    batch_size = training_config["batch_size"] if "batch_size" not in args.keys() else args["batch_size"]
    num_epochs = training_config["num_epochs"] if "num_epochs" not in args.keys() else args["num_epochs"]

    # Run name may be the same. So append current datetime to differentiate.
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")

    training_config_path = args["training_config"]
    training_config = json.load(open(training_config_path, "r"))

    args["disable_wandb"] = True if "disable_wandb" in args.keys() else False
    os.environ["WANDB_MODE"] = "offline" if args["disable_wandb"] else "online"
    project_name = args.get("project_name", "thesis")
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    model_config_paths = [os.path.join(args["model_config_dir"], f"{n}.json") for n in args["model_config_names"]]
    for cfg_path in model_config_paths:
        cfg_name = os.path.basename(cfg_path).split('.')[0] # Get config filename as config name.
        print(f"Training model with config {cfg_name}")
        runname = f"{args['run_name']}-{cfg_name}-b{batch_size}-e{num_epochs}-{cur_date}"

        save_dir = os.path.join("run", runname)
        print(f"Save Directory {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print("Initializing DNABERT-GSL")
        
        # Change model from DNABERT-SL to DNABERT-GSL with built in RNN.
        # model = init_seqlab_model(args["model_config"])
        model_config = json.load(open(cfg_path, "r"))
        bert = BertForMaskedLM.from_pretrained(str(PurePath(PureWindowsPath(model_config["pretrained"]))))
        bert = bert.bert
        model = DNABERT_GSL(bert, model_config)
        
        if not "mtl" in training_config.keys():
            print(">> Initializing default DNABERT-GSL.")
        else:
            if not training_config["mtl"] == "":
                print(f">> Initializing DNABERT-GSL with MTL {training_config['mtl']}")
                # Load BERT layer from MTL-trained folder.
                formatted_path = str(Path(PureWindowsPath(training_config["mtl"])))
                saved_model = BertForMaskedLM.from_pretrained(formatted_path)
                model.bert = saved_model.bert
            else:
                print(">> Invalid DNABERT-MTL result path. Initializing default DNABERT-GSL.")
    
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

        # Use scheduler from torch implementation for the sake of simplicity.
        scheduler = ExponentialLR(optimizer, gamma=0.1)

        # Freeze BERT layer if necessary.    
        if "freeze_bert" in training_config.keys():
            if training_config["freeze_bert"] > 0:
                print(">> Freeze BERT layer.", end="\r")
                for param in model.bert.parameters():
                    param.requires_grad = False
                print(f">> Freeze BERT layer. [{all([p.requires_grad == False for p in model.bert.parameters()])}]")

        
        # TODO: develop resume training feature here.
        training_counter = 0
        if "resume" in args.keys():
            resume_path = os.path.join(args["resume"])
            checkpoint_dir = os.path.basename(resume_path)
            last_epoch = checkpoint_dir.split('-')[1]
            training_counter = last_epoch + 1
            model.load_state_dict(
                torch.load(os.path.join(resume_path, "model.pth"))
            )
            optimizer.load_state_dict(
                torch.load(os.path.join(resume_path, "optimizer.pth"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(resume_path, "scheduler.pth"))
            )

        # Prepare wandb.
        wandb_cfg = {
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "device": device_name,
            "device_list": device_names,
            "training_counter": training_counter
        }
        print("Final Training Configuration")
        for key in wandb_cfg.keys():
            print(f"+ {key} {wandb_cfg[key]}")
        
        # run = wandb.init(project="thesis-mtl", entity="anwari32", config=wandb_cfg, reinit=True) 
        run = wandb.init(project=project_name, entity="anwari32", config=wandb_cfg, reinit=True) 
        if "run_name" in args.keys():
            wandb.run.name = f'{runname}-{wandb.run.id}'
            wandb.run.save()
        wandb.watch(model)
        
        # Save current model config in run folder.
        model_config_path = os.path.join(save_dir, "model_config.json")
        json.dump(model_config, open(model_config_path, "x"), indent=4)

        # Save current training config in run folder.
        training_config_path = os.path.join(save_dir, "training_config.json")
        json.dump(training_config, open(training_config_path, "x"), indent=4)

        print(f"Begin Training {wandb.run.name}")
        start_time = datetime.now()
        end_time = start_time
        try:
            trained_model, trained_optimizer, trained_scheduler = train(
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
                device_list=device_list,
                training_counter=training_counter
            )
        except Exception as ex:
            print(ex)
            end_time = datetime.now()
            running_time = end_time - start_time
            print(f"Error: Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")
            sys.exit(2)   

        end_time = datetime.now()
        running_time = end_time - start_time

        print(f"Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")

        total_config = {
            "training_config": training_config,
            "model_config": model_config,
            "start_time": start_time.strftime("%Y%m%d-%H%M%S"),
            "end_time": end_time.strftime("%Y%m%d-%H%M%S"),
            "running_time": str(running_time),
            "runname": args["run_name"]
        }
        training_info = os.path.join(save_dir, "training_info.json")
        json.dump(total_config, open(training_info, "x"), indent=4)
        run.finish()
        

