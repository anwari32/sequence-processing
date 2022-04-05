from genericpath import exists
from getopt import getopt
import json
import sys
import os
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sequential_labelling import preprocessing, train_using_genes
from utils.model import init_seqlab_model
from utils.optimizer import init_optimizer
import torch
from torch.cuda import device_count as cuda_device_count
from utils.utils import load_checkpoint, save_json_config
import wandb
import pandas as pd

def _parse_argv(argvs):
    opts, args = getopt(argvs, "t:m:d:f", ["training-config=", "model-config=", "device=", "force-cpu", "training-counter=", "resume-from-checkpoint=", "resume-from-optimizer=", "cuda-garbage-collection-mode="])
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
        elif o in ["--cuda-garbage-collection-mode"]:
            output["cuda_garbage_collection_mode"] = a
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

    training_config = json.load(open(args["training_config"], "r"))
    if not os.path.exists(training_config["result"]):
        os.makedirs(training_config["result"], exist_ok=True)
    if not os.path.exists(os.path.dirname(training_config["log"])):
        os.makedirs(os.path.dirname(training_config["log"]), exist_ok=True)

    model = init_seqlab_model(args["model_config"])
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

    if int(training_config["freeze_bert"]) > 0:
        for param in model.bert.parameters():
            param.requires_grad(False)

    loss_function = torch.nn.CrossEntropyLoss()

    gene_index = training_config["gene_index"]
    gene_dir = training_config["gene_dir"]

    df = pd.read_csv(training_config["gene_index"])
    train_genes = []
    for i, r in df.iterrows():
        train_genes.append(
            os.path.join(training_config["gene_dir"], r["chr"], r["gene"])
        )    
    # print("\n".join(train_genes))
    print(f"# Genes: {len(train_genes)}")
    training_steps = len(train_genes) * training_config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_config["warmup"], num_training_steps=training_steps)

    print(model)

    wandb.init(project="seqlab-training-by-genes", entity="anwari32")
    wandb.config = {
        "learning_rate": training_config["optimizer"]["learning_rate"],
        "epochs": training_config["num_epochs"],
        "batch_size": training_config["batch_size"]
    }

    model = train_using_genes(
        model=model, 
        tokenizer=BertTokenizer.from_pretrained(training_config["pretrained"]),
        optimizer=optimizer, 
        scheduler=scheduler, 
        train_genes=train_genes, 
        loss_function=loss_function, 
        num_epoch=training_config["num_epochs"], 
        batch_size=training_config["batch_size"], 
        grad_accumulation_steps=training_config["grad_accumulation_steps"],
        device=args["device"],
        wandb=wandb,
        training_counter=training_counter)

    total_config = {
        "training_config": training_config,
        "model_config": json.load(open(args["model_config"], "r")),
        "model_version": json.load(open(args["model_version"], "r")) if "model_version" in args.keys() else "default"
    }
    save_json_config(total_config, training_config["log"])
    
    

