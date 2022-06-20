from getopt import getopt
import sys
import json
from torch.optim import lr_scheduler
from torch.cuda import device_count as cuda_device_count, get_device_name
from torch.nn import BCELoss, CrossEntropyLoss
from utils.mtl import preprocessing_batches, preprocessing
from torch.optim import AdamW
import os
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
        "max-steps=",
        "fp16",
        "disable-wandb",
        "batch-sizes=",
        "num-epochs=",
        "do-kmer"
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
            output["run_name"] = [x for x in a.split(',')]
        elif o in ["--device-list"]:
            output["device_list"] = [int(x) for x in a.split(",")]
        elif o in ["--fp16"]:
            output["fp16"] = True
        elif o in ["--disable-wandb"]:
            output["disable_wandb"] = True
        elif o in ["--max-steps"]:
            output["max_steps"] = int(a)
        elif o in ["--batch-sizes"]:
            output["batch_sizes"] = [int(x) for x in a.split(',')]
        elif o in ["--num-epochs"]:
            output["num_epochs"] = int(a)
        elif o in ["--do-kmer"]:
            output["do_kmer"] = True
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
    assert "run_name" in args.keys(), f"Please specify runname. `--run-name=<runname>`"

    epoch_size = training_config["num_epochs"] if "num_epochs" not in args.keys() else args["num_epochs"] # Override num epochs if given in command.
    batch_sizes = [training_config["batch_size"]] if "batch_sizes" not in args.keys() else args["batch_sizes"] # Override batch size if given in command.

    # Since we have opened possible multiple batch sizes, each of run names must be correlated with each of batch sizes.
    run_names = args["run_name"]
    if len(run_names) != len(batch_sizes):
        raise ValueError(f"`run names` do not correspond to `batch sizes`.")
    
    # Run name may be the same. So append current datetime to differentiate.
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    # args["run_name"] = f"{args['run_name']}-{cur_date}"
    run_names = [f"{n}-{cur_date}" for n in run_names]

    # print(model)
    print(f"Preparing Training Data")
    if "do_kmer" not in args.keys():
        args["do_kmer"] = False
    
    dataloaders = None
    if len(batch_sizes) == 0:
        raise ValueError(f"Length batch size must not be 0.")
    dataloaders = preprocessing_batches(
        training_config["train_data"],  # CSV file,
        training_config["pretrained"],  # Pretrained_path,
        batch_sizes=batch_sizes,        # Batch size,
        do_kmer=args["do_kmer"]   
    )
    
    print(f"Preparing Validation Data")
    validation_dataloader = None
    if "validation_data" in training_config.keys():
        eval_data_path = str(Path(PureWindowsPath(training_config["validation_data"])))
        validation_dataloader = preprocessing(eval_data_path, training_config["pretrained"], 1, do_kmer=args["do_kmer"])

    loss_fn = {
        "prom": BCELoss() if training_config["prom_loss_fn"] == "bce" else CrossEntropyLoss(),
        "ss": BCELoss() if training_config["ss_loss_fn"] == "bce" else CrossEntropyLoss(),
        "polya": BCELoss() if training_config["polya_loss_fn"] == "bce" else CrossEntropyLoss()
    }

    # Enable or disable wandb real time sync.
    args["disable_wandb"] = True if "disable_wandb" in args.keys() else False
    os.environ["WANDB_MODE"] = "offline" if args["disable_wandb"] else "online"

    for run_name, batch_size, dataloader in zip(run_names, batch_sizes, dataloaders):
        print(f"Runname {run_name}, Batch size {batch_size}")

        print(f"Preparing Model & Optimizer")
        model = init_mtl_model(args["model_config"])
        model.to(args["device"])
        optimizer = AdamW(model.parameters(), 
            lr=training_config["optimizer"]["learning_rate"], 
            betas=(training_config["optimizer"]["beta1"], training_config["optimizer"]["beta1"]),
            eps=training_config["optimizer"]["epsilon"],
            weight_decay=training_config["optimizer"]["weight_decay"]
        )

        training_steps = len(dataloader) * epoch_size
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)
        # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

        # log_dir_path = str(Path(PureWindowsPath(training_config["log"])))
        # log_file_path = os.path.join(log_dir_path, cur_date, "log.csv")
        log_file_path = os.path.join("run", run_name, "logs", "log.csv")

        # save_model_path = str(Path(PureWindowsPath(training_config["result"])))
        # save_model_path = os.path.join(save_model_path, cur_date)
        save_model_path = os.path.join("run", run_name)

        for p in [log_file_path, save_model_path]:
            os.makedirs(os.path.dirname(p), exist_ok=True)

        # Save current model config in run folder.
        model_config = json.load(open(str(Path(PureWindowsPath(args["model_config"]))), "r"))
        json.dump(model_config, open(os.path.join("run", run_name, "model_config.json"), "x"), indent=4)

        start_time = datetime.now()
        trained_model, trained_optimizer = None, None
        save_dir = os.path.join("run", run_name)
        device = args["device"]
        training_counter = args["training_counter"] if "training_counter" in args.keys() else 0
        loss_strategy=training_config["loss_strategy"]
        grad_accumulation_steps=training_config["grad_accumulation_steps"]
        device_list=args["device_list"] if "device_list" in args.keys() else []

        # Prepare wandb.
        device_name = get_device_name(device)
        device_names = ", ".join([get_device_name(f"cuda:{a}") for a in device_list])
        wandb_cfg = {
            "learning_rate": training_config["optimizer"]["learning_rate"],
            "epochs": epoch_size,
            "batch_size": batch_size,
            "device": device_name,
            "device_list": device_names,
        }
        print("Final Training Configuration")
        for key in wandb_cfg.keys():
            print(f"{key} {wandb_cfg[key]}")

        wandb.init(project="thesis-mtl", entity="anwari32", config=wandb_cfg) 
        if "run_name" in args.keys():
            wandb.run.name = f'{run_name}-{wandb.run.id}'
            wandb.run.save()
        wandb.watch(model)

        if "max_steps" in args.keys():
            from multitask_learning import train_by_steps

            # Scheduler must be re-initialized because epochs is determined by max_steps.
            # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=args["max_steps"])
            trained_model, trained_optimizer = train_by_steps(
                dataloader, 
                model, 
                loss_fn, 
                optimizer, 
                scheduler, 
                args["max_steps"], 
                batch_size,
                save_dir, 
                device, 
                training_counter, 
                loss_strategy, 
                grad_accumulation_steps, 
                wandb, 
                validation_dataloader, 
                device_list)
        else:
            from multitask_learning import train

            trained_model, trained_optimizer, trained_scheduler = train(
                dataloader, 
                model, 
                loss_fn, 
                optimizer, 
                scheduler, 
                batch_size=batch_size, 
                epoch_size=epoch_size, 
                device=args["device"], 
                save_dir=save_model_path, 
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
        save_json_config(total_config, os.path.join("run", run_name, "final_config.json"))

        # Save final trained model.
        save_checkpoint(trained_model, trained_optimizer, trained_scheduler, total_config, os.path.join(save_dir, "final-checkpoint.pth"))
    
