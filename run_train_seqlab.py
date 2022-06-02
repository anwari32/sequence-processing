from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count
from sequential_labelling import train
from utils.seqlab import preprocessing
from utils.model import init_seqlab_model
from utils.optimizer import init_optimizer
from utils.utils import load_checkpoint
from transformers import BertTokenizer
import os
import wandb
from datetime import datetime
from pathlib import Path, PureWindowsPath
from utils.utils import save_json_config, save_checkpoint
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler

def parse_args(argv):
    opts, args = getopt(argv, "t:m:d:f", ["training-config=", 
        "model-config=", 
        "device=", 
        "force-cpu", 
        "training-counter=", 
        "device-list=", 
        "run-name=", 
        "disable-wandb",
        "batch-size=",
        "num-epochs="
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
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    for key in args.keys():
        print(key, args[key])

    # Make sure input parameters are valid.
    if not os.path.exists(args["training_config"]) or not os.path.isfile(args["model_config"]):
        print(f"Training config not found at {args['training_config']}")
    
    if not os.path.exists(args["model_config"]) or not os.path.isfile(args["model_config"]):
        print(f"Model config not found at {args['model_config']}")

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

    # Run name may be the same. So append current datetime to differentiate.
    # Create this folder if not exist.
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    args["run_name"] = f"{args['run_name']}-{cur_date}"

    training_config_path = str(Path(PureWindowsPath(args["training_config"])))
    training_config = json.load(open(training_config_path, "r"))
    training_filepath = str(Path(PureWindowsPath(training_config["train_data"])))
    validation_filepath = str(Path(PureWindowsPath(training_config["validation_data"])))
    print(f"Preparing Training Data {training_filepath}")
    dataloader = preprocessing(
        training_filepath,# csv_file, 
        BertTokenizer.from_pretrained(str(Path(PureWindowsPath(training_config["pretrained"])))), # tokenizer
        training_config["batch_size"], #batch_size,
        do_kmer=False
        )
    print(f"Preparing Validation Data {validation_filepath}")
    eval_dataloader = preprocessing(
        validation_filepath,# csv_file, 
        BertTokenizer.from_pretrained(str(Path(PureWindowsPath(training_config["pretrained"])))), # tokenizer
        1,
        do_kmer=False
    )

    model = init_seqlab_model(args["model_config"])

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

    if int(training_config["freeze_bert"]) > 0:
        for param in model.bert.parameters():
            param.requires_grad(False)

    # Override batch size and epoch if given in command.
    epoch_size = training_config["num_epochs"] if "num_epochs" not in args.keys() else args["num_epochs"]
    batch_size = training_config["batch_size"] if "batch_size" not in args.keys() else args["batch_size"]
    
    training_steps = len(dataloader) * epoch_size
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # Prepare save directory for this work.
    save_dir = os.path.join("run", args["run_name"])
    print(f"Save Directory {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Save current model config in run folder.
    model_config = json.load(open(str(Path(PureWindowsPath(args["model_config"]))), "r"))
    model_config_path = os.path.join("run", args["run_name"], "model_config.json")
    json.dump(model_config, open(model_config_path, "x"), indent=4)
    
    loss_function = CrossEntropyLoss()

    start_time = datetime.now()
    trained_model, optimizer, scheduler = train(
        model, 
        optimizer, 
        scheduler, 
        dataloader, 
        epoch_size, 
        loss_function,
        save_dir,
        eval_dataloader,
        device=args["device"], 
        device_list=(args["device_list"] if "device_list" in args.keys() else []),
        wandb=wandb,
        loss_strategy="sum"
    )
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

    save_checkpoint(model, optimizer, scheduler, total_config, save_dir)