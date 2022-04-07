from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count
from torch.nn import BCELoss, CrossEntropyLoss
from multitask_learning import train, preprocessing
from utils.optimizer import init_optimizer
from transformers import get_linear_schedule_with_warmup
import os
from data_dir import pretrained_3kmer_dir
from utils.model import init_mtl_model
from pathlib import Path, PureWindowsPath

import wandb

def parse_args(argv):
    opts, args = getopt(argv, "t:m:d:f", ["training-config=", "model-config=", "device=", "force-cpu", "training-counter=", "resume-from-checkpoint=", "resume-from-optimizer=", "cuda-garbage-collection-mode="])
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
        elif o in ["--resume-from-optimizer"]:
            output["resume_from_optimimzer"] = a
        elif o in ["--cuda-garbage-collection-mode"]:
            output["cuda_garbage_collection_mode"] = a
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    for key in args.keys():
        print(key, args[key])

    # Make sure input parameters are valid.
    if not "force-cpu" in args.keys():
        if args["device"] == "cpu":
            print(f"Don't use CPU for training")
            sys.exit(2)
        cuda_device_count = cuda_device_count()
        if cuda_device_count > 1 and args["device"] == "cuda":
            print(f"There are more than one CUDA devices. Please choose one.")
            sys.exit(2)
    
    model = init_mtl_model(args["model_config"])
    print(model)

    training_config = json.load(open(args["training_config"], 'r'))
    dataloader = preprocessing(
        training_config["train_data"],# csv_file, 
        training_config["pretrained"], #pretrained_path, 
        training_config["batch_size"], #batch_size
        )

    loss_fn = {
        "prom": BCELoss() if training_config["prom_loss_fn"] == "bce" else CrossEntropyLoss(),
        "ss": BCELoss() if training_config["ss_loss_fn"] == "bce" else CrossEntropyLoss(),
        "polya": BCELoss() if training_config["polya_loss_fn"] == "bce" else CrossEntropyLoss()
    }

    optimizer = init_optimizer(
        training_config["optimizer"]["name"], 
        model.parameters(), 
        training_config["optimizer"]["learning_rate"], 
        training_config["optimizer"]["epsilon"], 
        training_config["optimizer"]["beta1"], 
        training_config["optimizer"]["beta2"], 
        training_config["optimizer"]["weight_decay"]
    )

    epoch_size = training_config["num_epochs"]
    batch_size = training_config["batch_size"]
    
    training_steps = len(dataloader) * epoch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)
    
    log_file_path = str(Path(PureWindowsPath(training_config["log"])))
    save_model_path = str(Path(PureWindowsPath(training_config["result"])))
    for p in [log_file_path, save_model_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    wandb.init(project="thesis-mtl", entity="anwari32")    
    wandb.config = {
        "learning_rate": training_config["optimizer"]["learning_rate"],
        "epochs": training_config["num_epochs"],
        "batch_size": training_config["batch_size"]
    }

    whole_config = {
        "model": json.load(open(str(Path(PureWindowsPath(args["model_config"]))), "r")),
        "training": training_config
    }

    trained_model = train(
        dataloader, 
        model, 
        loss_fn, 
        optimizer, 
        scheduler, 
        batch_size=training_config["batch_size"], 
        epoch_size=training_config["num_epochs"], 
        log_file_path=log_file_path, 
        device=args["device"], 
        save_model_path=save_model_path, 
        remove_old_model=False, 
        training_counter=args["training_counter"] if "training_counter" in args.keys() else 0, 
        loss_strategy=training_config["loss_strategy"], 
        grad_accumulation_steps=training_config["grad_accumulation_steps"], 
        #resume_from_checkpoint=args["resume_from_checkpoint"] if "resume_from_checkpoint" in args.keys() else None, 
        #resume_from_optimizer=args["resume_from_optimizer"] if "resume_from_optimizer" in args.keys() else None,
        wandb=wandb,
    )
