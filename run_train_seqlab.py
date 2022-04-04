from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count
from sequential_labelling import train
from utils.seqlab import preprocessing
from utils.model import init_seqlab_model
from utils.optimizer import init_optimizer
from utils.utils import load_checkpoint
from transformers import get_linear_schedule_with_warmup, BertTokenizer
import os
import wandb

def parse_args(argv):
    opts, args = getopt(argv, "t:m:d:f", ["config=", "device=", "force-cpu", "training-counter=", "resume-from-checkpoint=", "resume-from-optimizer="])
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
    
    config = json.load(open(args["training_config"], 'r'))
    dataloader = preprocessing(
        config["train_data"],# csv_file, 
        BertTokenizer.from_pretrained(config["pretrained"]), #pretrained_path, 
        config["batch_size"], #batch_size,
        do_kmer=True
        )

    model = init_seqlab_model(args["model_config"])

    optimizer = init_optimizer(
        config["optimizer"]["name"], 
        model.parameters(), 
        config["optimizer"]["learning_rate"], 
        config["optimizer"]["epsilon"], 
        config["optimizer"]["beta1"], 
        config["optimizer"]["beta2"], 
        config["optimizer"]["weight_decay"]
    )

    training_counter = args["training_counter"] if "training_counter" in args.keys() else 0
    if "resume_from_checkpoint" in config.keys() and config["resume_from_checkpoint"] != "":
        checkpoint = load_checkpoint(config["resume_from_checkpoint"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        config = checkpoint["config"]
        training_counter = int(config['epoch']) + 1
        print(f"Resuming training from epoch {training_counter}") 

    if int(config["freeze_bert"]) > 0:
        for param in model.bert.parameters():
            param.requires_grad(False)

    epoch_size = config["num_epochs"]
    batch_size = config["batch_size"]
    
    training_steps = len(dataloader) * epoch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup"], num_training_steps=training_steps)
    
    log_file_path = config["log"]
    save_model_path = config["result"]
    for p in [log_file_path, save_model_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    wandb.init(project="seqlab-training-by-sequence", entity="anwari32")
    wandb.config = {
        "learning_rate": config["optimizer"]["learning_rate"],
        "epochs": config["num_epochs"],
        "batch_size": config["batch_size"]
    }

    trained_model = train(model, 
        optimizer, 
        scheduler, 
        dataloader, 
        epoch_size, 
        batch_size, 
        log_path=log_file_path, 
        save_model_path=save_model_path, 
        device=args["device"], 
        remove_old_model=False, 
        training_counter=training_counter, 
        grad_accumulation_steps=config["grad_accumulation_steps"], 
        loss_strategy=config["loss_strategy"],
        wandb=wandb)
