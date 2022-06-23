from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count
from sequential_labelling import evaluate_sequences
from utils.seqlab import preprocessing
from utils.model import init_seqlab_model
from utils.utils import load_checkpoint, save_json_config
from transformers import BertTokenizer, BertForMaskedLM
import os
import wandb

def parse_args(argv):
    opts, args = getopt(argv, "w:e:d:", ["work-dir=", "eval-data=", "device="])
    output = {}
    for o, a in opts:
        if o in ["-w", "--work-dir"]:
            output["workdir"] = a
        elif o in ["-e", "--eval-data"]:
            output["eval_data"] = a
        elif o in ["-d", "--device"]:
            output["device"] = True
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    for key in args.keys():
        print(key, args[key])

    dataloader = preprocessing(
        args["eval_data"],  # csv_file, 
        BertTokenizer.from_pretrained(config["pretrained"]), #pretrained_path, 
        1, #batch_size,
        do_kmer=False
    )

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
        model_checkpoint = os.path.join(dpath, "model.pth")
        optimizer_checkpoint = os.path.join(dpath, "optimizer.pth")
        scheduler_checkpoint = os.path.join(dpath, "scheduler.pth")
        log_path = os.path.join(dpath, "validation_log.csv")

        bert = BertForMaskedLM.from_pretrained(str(Path(PureWindowsPath("pretrained\\3-new-12w-0")))).bert
        model = DNABERT_SL(bert, None)
        model.load_state_dict(model_checkpoint)
        
        wandb.init(project="thesis", entity="anwari32", config={
            "epoch": epoch
        })
        wandb.define_metric("epoch")
        wandb.define_metric("average_accuracy", "epoch")
        wandb.define_metric("average_loss", "epoch")

        avg_acc, avg_loss = evaluate_sequences(model, dataloader, device, log_path, epoch, num_epoch, loss_fn, loss_strategy)

        wandb.log({
            "average_accuracy": avg_acc,
            "average_loss": avg_loss,
            "epoch": epoch
        })
