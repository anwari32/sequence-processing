from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count
from sequential_gene_labeling import evaluate
from utils.seqlab import preprocessing
from utils.model import init_seqlab_model
from utils.utils import load_checkpoint, save_json_config
from transformers import BertTokenizer, BertForMaskedLM
import os
import wandb
import pandas as pd

def parse_args(argv):
    opts, args = getopt(argv, "w:e:d:m:", ["work-dir=", "eval-data=", "device=", "model-config="])
    output = {}
    for o, a in opts:
        if o in ["-w", "--work-dir"]:
            output["workdir"] = a
        elif o in ["-e", "--eval-data"]:
            output["eval_data"] = a
        elif o in ["-d", "--device"]:
            output["device"] = True
        elif o in ["-m", "--model-config"]:
            output["model_config"] = a
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    for key in args.keys():
        print(key, args[key])

    assert "workdir" in args.keys(), f"`workdir` not specified. Workdir folder contains folders in each which contains all model checkpoints."
    assert "eval_data" in args.keys(), f"`eval_data` not specified."
    assert "device" in args.keys(), f"`device` not specified"

    eval_genes_df = pd.read_csv(args["eval_data"])
    eval_genes_dir = os.path.dirname(args["eval_data"])
    eval_genes = [os.path.join(eval_genes_dir, r["chr"], r["gene"]) for i, r in eval_genes_df.iterrows()]

    device = args["device"]

    # Workdir contains folder in which model for every epoch is saved.
    # Traverse thorough that folder.
    directories = os.listdir(args["workdir"])
    num_epoch = len(directories)
    loss_fn = CrossEntropyLoss()
    loss_strategy = "sum"

    # for d in directories:
    for idx in range(num_epoch):
        d = directories[idx] 
        epoch = d.split("-")[1] # Get epoch number from dir.
        dpath = os.path.join(args["workdir"], d)

        mpath = os.path.join(dpath, "model.pth")
        optimpath = os.path.join(dpath, "optimizer.pth")
        schpath = os.path.join(dpath, "scheduler.pth")
        cfgpath = os.path.join(dpath, "model_config.json")

        # If model config is specified, use it instead of model config found in folder.
        if "model_config" in args.keys():
            cfg_path = args["model_config"]

        assert os.path.exists(mpath), f"Model not found at {mpath}"
        assert os.path.exists(optimpath), f"Optimizer not found at {optimpath}"
        assert os.path.exists(schpath), f"Scheduler not found at {schpath}"
        assert os.path.exists(cfgpath), f"Model config not found at {cfgpath}"
        
        log_path = os.path.join(dpath, "validation_log.csv")

        bert = BertForMaskedLM.from_pretrained(str(Path(PureWindowsPath("pretrained\\3-new-12w-0")))).bert
        cfg = json.load(open(cfgpath, "r"))

        model = DNABERT_SL(bert, cfg)
        model.load_state_dict(mpath)
        
        wandb.init(project="thesis-sequential-gene-labelling", entity="anwari32", config={
            "epoch": epoch
        })
        wandb.define_metric("epoch")
        wandb.define_metric("average_gene_accuracy", "epoch")
        wandb.define_metric("average_gene_loss", "epoch")

        avg_accuracy_score, avg_incorrect_score, avg_gene_loss_score = evaluate(model, eval_genes, device, eval_log, epoch, num_epoch, loss_fn, wandb)

        wandb.log({
            "average_gene_accuracy": avg_accuracy_score,
            "average_gene_loss": avg_gene_loss_score,
            "epoch": epoch
        })
