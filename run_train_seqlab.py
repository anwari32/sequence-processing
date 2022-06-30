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
from utils.utils import save_checkpoint
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler

def parse_args(argv):
    opts, args = getopt(argv, "t:m:d:f:r", [
        "training-config=", 
        "model-config=", 
        "device=", 
        "force-cpu", 
        "training-counter=", 
        "device-list=", 
        "run-name=", 
        "disable-wandb",
        "batch-size=",
        "num-epochs=",
        "resume=",
        "loss-strategy=",
        "model-config-dir=",
        "model-config-names="
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
        elif o in ["-r", "--resume"]:
            output["resume"] = a
        elif o in ["--loss-strategy"]:
            output["loss_strategy"] = a
        elif o in ["--model-config-dir"]:
            output["model_config_dir"] = a
        elif o in ["--model-config-names"]:
            output["model_config_names"] = a.split(",")
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

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

    # Override batch size and epoch if given in command.
    epoch_size = training_config["num_epochs"] if "num_epochs" not in args.keys() else args["num_epochs"]
    batch_size = training_config["batch_size"] if "batch_size" not in args.keys() else args["batch_size"]

    training_config_path = args["training_config"]
    training_config = json.load(open(training_config_path, "r"))
    training_filepath = str(Path(PureWindowsPath(training_config["train_data"])))
    validation_filepath = str(Path(PureWindowsPath(training_config["validation_data"])))
    print(f"Preparing Training Data {training_filepath}")
    dataloader = preprocessing(
        training_filepath,# csv_file, 
        BertTokenizer.from_pretrained(str(Path(PureWindowsPath(training_config["pretrained"])))), # tokenizer
        batch_size, # training_config["batch_size"], #batch_size,
        do_kmer=False
        )
    print(f"Preparing Validation Data {validation_filepath}")
    eval_dataloader = preprocessing(
        validation_filepath,# csv_file, 
        BertTokenizer.from_pretrained(str(Path(PureWindowsPath(training_config["pretrained"])))), # tokenizer
        1,
        do_kmer=False
    )

    # All training devices are CUDA GPUs.
    device_name = get_device_name(args["device"])
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    for cfg_path in model_config_list:
        cfg_name = os.path.basename(cfg_path).split(".")[0] # Get filename without extension.
        print(f"Training model with config {cfg_name}")
        runname = f"{args['run_name']}-{cfg_name}-b{batch_size}-e{epoch_size}-{cur_date}"
        
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
        training_counter = 0

        # TODO: develop resume training feature here.
        # Resume training of checkpoint is stated.
        # For now, it only works in single config.
        if "resume" in args.keys():
            resume_path = os.path.join(args["resume"])
            checkpoint_dir = os.path.basename(resume_path)
            last_epoch = checkpoint_dir.split("-")[1]
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
            print(f"Continuing training. Start from epoch {training_counter}")

        if int(training_config["freeze_bert"]) > 0:
            print("Freezing BERT")
            for param in model.bert.parameters():
                param.requires_grad(False)

        # Prepare save directory for this work.
        save_dir = os.path.join("run", runname)
        print(f"Save Directory {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
        # Save current model config in run folder.
        model_config = json.load(open(cfg_path, "r"))
        model_config_path = os.path.join(save_dir, "model_config.json")
        json.dump(model_config, open(model_config_path, "x"), indent=4)

        # Save current training config in run folder.
        training_config_path = os.path.join(save_dir, "training_config.json")
        json.dump(training_config, open(training_config_path, "x"), indent=4)
    
        # Loss function.
        loss_function = CrossEntropyLoss()

        # Loss strategy.
        loss_strategy = "sum" # Default mode.
        if "loss_strategy" in args.keys():
            loss_strategy = args["loss_strategy"]

        # Final training configuration.
        tcfg = {
            "training_data": training_filepath,
            "validation_data": validation_filepath,
            "learning_rate": lr,
            "epochs": epoch_size,
            "batch_size": batch_size,
            "device": device_name,
            "device_list": device_names,
            "loss_strategy": loss_strategy
        }
        print("Final Training Configuration")
        for k in tcfg.keys():
            print(f"+ {k} {tcfg[k]}")

        # Prepare wandb.
        run = wandb.init(project="thesis-mtl", entity="anwari32", config=tcfg, reinit=True) 
        if "run_name" in args.keys():
            wandb.run.name = f'{runname}-{wandb.run.id}'
            wandb.run.save()
        wandb.watch(model)

        print(f"Begin Training {wandb.run.name}")
        start_time = datetime.now()
        trained_model, trained_optimizer, trained_scheduler = train(
            model, 
            optimizer, 
            scheduler, 
            dataloader, 
            epoch_size, 
            save_dir,
            loss_function,
            device=args["device"], 
            loss_strategy=loss_strategy,
            wandb=wandb,
            device_list=device_list,
            eval_dataloader=eval_dataloader,    
            training_counter=training_counter    
        )
        end_time = datetime.now()
        running_time = end_time - start_time

        print(f"Start Time {start_time}\nFinish Time {end_time}\nTraining Duration {running_time}")

        total_config = {
            "training_config": training_config,
            "model_config": model_config,
            "start_time": start_time.strftime("%Y%m%d-%H%M%S"),
            "end_time": end_time.strftime("%Y%m%d-%H%M%S"),
            "running_time": str(running_time),
            "runname": runname
        }
        training_info_path = os.path.join(save_dir, "training_info.json")
        json.dump(total_config, open(training_info_path, "x"), indent=4)
        run.finish()