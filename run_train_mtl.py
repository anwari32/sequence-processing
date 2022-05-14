from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count
from torch.nn import BCELoss, CrossEntropyLoss
from multitask_learning import train, preprocessing
from utils.optimizer import init_optimizer
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import os
from data_dir import pretrained_3kmer_dir
from utils.model import init_mtl_model
from utils.utils import save_checkpoint, save_json_config
from pathlib import Path, PureWindowsPath
from datetime import datetime

import wandb

def parse_args(argv):
    opts, args = getopt(argv, "t:m:d:f", [
        "training-config=", 
        "model-config=", 
        "device=", 
        "force-cpu", 
        "training-counter=", 
        "resume-from-checkpoint=", 
        "resume-from-optimizer=", 
        "cuda-garbage-collection-mode=", 
        "run-name=",
        "device-list=",
        "fp16",
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
        elif o in ["--resume-from-optimizer"]:
            output["resume_from_optimimzer"] = a
        elif o in ["--cuda-garbage-collection-mode"]:
            output["cuda_garbage_collection_mode"] = a
        elif o in ["--run-name"]:
            output["run_name"] = a
        elif o in ["--device-list"]:
            output["device_list"] = [int(x) for x in a.split(",")]
        elif o in ["--fp16"]:
            output["fp16"] = True
        elif o in ["--disable-wandb"]:
            output["disable_wandb"] = True
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    for key in args.keys():
        print(key, args[key])
    
    training_config = json.load(open(args["training_config"], 'r'))

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
        print("Please specify runname.")
        print("`--run-name=<runname>`")
        sys.exit(2)
    
    # Run name may be the same. So append current datetime to differentiate.
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    args["run_name"] = f"{args['run_name']}-{cur_date}"

    print(f"Preparing Model & Optimizer")
    model = init_mtl_model(args["model_config"])
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

    # print(model)
    print(f"Preparing Training Data")
    dataloader = preprocessing(
        training_config["train_data"],# csv_file, 
        training_config["pretrained"], #pretrained_path, 
        training_config["batch_size"], #batch_size
        )
    
    print(f"Preparing Validation Data")
    validation_dataloader = None
    if "validation_data" in training_config.keys():
        eval_data_path = str(Path(PureWindowsPath(training_config["validation_data"])))
        validation_dataloader = preprocessing(eval_data_path, training_config["pretrained"], 1)

    loss_fn = {
        "prom": BCELoss() if training_config["prom_loss_fn"] == "bce" else CrossEntropyLoss(),
        "ss": BCELoss() if training_config["ss_loss_fn"] == "bce" else CrossEntropyLoss(),
        "polya": BCELoss() if training_config["polya_loss_fn"] == "bce" else CrossEntropyLoss()
    }

    epoch_size = training_config["num_epochs"]
    batch_size = training_config["batch_size"]
    
    training_steps = len(dataloader) * epoch_size
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)

    # log_dir_path = str(Path(PureWindowsPath(training_config["log"])))
    # log_file_path = os.path.join(log_dir_path, cur_date, "log.csv")
    log_file_path = os.path.join("run", args["run_name"], "logs", "log.csv")

    # save_model_path = str(Path(PureWindowsPath(training_config["result"])))
    # save_model_path = os.path.join(save_model_path, cur_date)
    save_model_path = os.path.join("run", args["run_name"])

    for p in [log_file_path, save_model_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    if "disable_wandb" not in args.keys():
        args["disable_wandb"] = False
    if not args["disable_wandb"]:    
        wandb.init(project="thesis-mtl", entity="anwari32") 
        if "run_name" in args.keys():
            wandb.run.name = f'{args["run_name"]}-{wandb.run.id}'
            wandb.run.save()
        wandb.config = {
            "learning_rate": training_config["optimizer"]["learning_rate"],
            "epochs": training_config["num_epochs"],
            "batch_size": training_config["batch_size"]
        }
        wandb.watch(model)

    # Save current model config in run folder.
    model_config = json.load(open(str(Path(PureWindowsPath(args["model_config"]))), "r"))
    json.dump(model_config, open(os.path.join("run", args["run_name"], "model_config.json"), "x"), indent=4)

    start_time = datetime.now()
    trained_model, trained_optimizer = train(
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
        training_counter=args["training_counter"] if "training_counter" in args.keys() else 0, 
        loss_strategy=training_config["loss_strategy"], 
        grad_accumulation_steps=training_config["grad_accumulation_steps"], 
        wandb=wandb,
        eval_dataloader=validation_dataloader,
        device_list=args["device_list"] if "device_list" in args.keys() else [],
    )
    end_time = datetime.now()
    running_time = end_time - start_time

    print(f"Training Duration {running_time}")

    total_config = {
        "training_config": training_config,
        "model_config": json.load(open(str(Path(PureWindowsPath(args["model_config"]))), "r")),
        "start_time": start_time.strftime("%Y%m%d-%H%M%S"),
        "end_time": end_time.strftime("%Y%m%d-%H%M%S"),
        "running_time": str(running_time),
        "runname": args["run_name"]
    }

    # save_json_config(total_config, os.path.join(os.path.dirname(str(Path(PureWindowsPath(training_config["log"])))), "config.json"))
    save_json_config(total_config, os.path.join("run", args["run_name"], "final_config.json"))

    # Save final trained model.
    save_checkpoint(trained_model, trained_optimizer, total_config, os.path.join(save_model_path, "final-checkpoint.pth"))
    
