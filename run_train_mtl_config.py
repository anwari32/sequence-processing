from getopt import getopt
from mimetypes import init
import sys
import json
from torch.cuda import device_count as cuda_device_count
from torch.nn import BCELoss, CrossEntropyLoss
from multitask_learning import train, preprocessing, init_model_mtl
from utils.optimizer import init_optimizer
from transformers import get_linear_schedule_with_warmup
import os

import wandb

def parse_args(argv):
    opts, args = getopt(argv, "c:d:f", ["config=", "device=", "force-cpu", "training-counter=", "resume-from-checkpoint=", "resume-from-optimizer="])
    output = {}
    for o, a in opts:
        if o in ["-c", "--config"]:
            output["config"] = a
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
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    # Make sure input parameters are valid.
    if not "force-cpu" in args.keys():
        if args["device"] == "cpu":
            print(f"Don't use CPU for training")
            sys.exit(2)
        cuda_device_count = cuda_device_count()
        if cuda_device_count > 1 and args["device"] == "cuda":
            print(f"There are more than one CUDA devices. Please choose one.")
            sys.exit(2)
    
    config = json.load(open(args["config"], 'r'))
    dataloader = preprocessing(
        config["train_data"],# csv_file, 
        config["pretrained"], #pretrained_path, 
        config["batch_size"], #batch_size
        )

    model = init_model_mtl(config["pretrained"], json.load(open(config["arch"], "r")))

    loss_fn = {
        "prom": BCELoss(),
        "ss": CrossEntropyLoss(),
        "polya": CrossEntropyLoss()
    }

    optimizer = init_optimizer(
        config["optimizer"]["name"], 
        model.parameters(), 
        config["optimizer"]["learning_rate"], 
        config["optimizer"]["epsilon"], 
        config["optimizer"]["beta1"], 
        config["optimizer"]["beta2"], 
        config["optimizer"]["weight_decay"]
    )

    epoch_size = config["num_epochs"]
    batch_size = config["batch_size"]
    
    training_steps = len(dataloader) * epoch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup"], num_training_steps=training_steps)
    
    log_file_path = config["log"]
    save_model_path = config["result"]
    for p in [log_file_path, save_model_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    wandb.init(project="thesis-mtl", entity="anwari32")    
    wandb.config = {
        "learning_rate": config["optimizer"]["learning_rate"],
        "epochs": config["num_epochs"],
        "batch_size": config["batch_size"]
    }

    trained_model = train(
        dataloader, 
        model, 
        loss_fn, 
        optimizer, 
        scheduler, 
        batch_size=config["batch_size"], 
        epoch_size=config["num_epochs"], 
        log_file_path=log_file_path, 
        device=args["device"], 
        save_model_path=save_model_path, 
        remove_old_model=False, 
        training_counter=args["training_counter"] if "training_counter" in args.keys() else 0, 
        loss_strategy=config["loss_strategy"], 
        grad_accumulation_steps=config["grad_accumulation_steps"], 
        resume_from_checkpoint=args["resume_from_checkpoint"] if "resume_from_checkpoint" in args.keys() else None, 
        resume_from_optimizer=args["resume_from_optimizer"] if "resume_from_optimizer" in args.keys() else None,
        wandb=wandb
    )
