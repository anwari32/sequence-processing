from getopt import getopt
import sys
import json
from torch.cuda import device_count as cuda_device_count
from sequential_labelling import do_evaluate
from utils.seqlab import preprocessing
from utils.model import init_seqlab_model
from utils.utils import load_checkpoint, save_json_config
from transformers import BertTokenizer
import os
import wandb

def parse_args(argv):
    opts, args = getopt(argv, "e:d:m:v:f", ["eval-config=", "device=", "model-config=", "model-version="])
    output = {}
    for o, a in opts:
        if o in ["-e", "--eval-config"]:
            output["eval_config"] = a
        elif o in ["-d", "--device"]:
            output["device"] = a
        elif o in ["-f", "--force-cpu"]:
            output["force-cpu"] = True
        elif o in ["-m", "--model-config"]:
            output["model_config"]= a
        elif o in ["-v", "--model-version"]:
            output["model_version"] = a
        else:
            print(f"Argument {o} not recognized.")
            sys.exit(2)
    return output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    for key in args.keys():
        print(key, args[key])

    # Make sure input parameters are valid.
    # Make sure input parameters are valid.
    if not "force-cpu" in args.keys():
        if args["device"] == "cpu":
            print(f"Don't use CPU for training")
            sys.exit(2)
        cuda_device_count = cuda_device_count()
        if cuda_device_count > 1 and args["device"] == "cuda":
            print(f"There are more than one CUDA devices. Please choose one.")
            sys.exit(2)
    
    config = json.load(open(args["eval_config"], 'r'))
    dataloader = preprocessing(
        config["eval_data"],# csv_file, 
        BertTokenizer.from_pretrained(config["pretrained"]), #pretrained_path, 
        config["batch_size"], #batch_size,
        do_kmer=True
    )

    model = init_seqlab_model(args["model_config"])
    if "model_version" in args.keys():
        checkpoint = load_checkpoint(args["model_version"])
        model.load_state_dict(checkpoint["model"])
    else:
        print("Evaluating default model.")
    model.to(args["device"])
    model.eval()
    
    if os.path.exists(config["log"]):
        os.remove(config["log"])
    else:
        os.makedirs(os.path.dirname(config["log"]), exist_ok=True)

    wandb.init(project="seqlab-eval-by-sequence", entity="anwari32")

    complete_config = {
        "eval_config": config,
        "model_config": json.load(open(args["model_config"], "r")),
        "model_version": json.load(open(args["model_version"], "r")) if "model_version" in args.keys() else "default"
    }

    result = do_evaluate(model, dataloader, log=config["log"], device=args["device"])
    save_json_config(complete_config, os.path.join(os.path.dirname(config["log"]), "evaluation_config.json"))
    print(result)
