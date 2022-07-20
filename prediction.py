import sys
import os
from getopt import getopt
from models.seqlab import DNABERT_SL
from utils.seqlab import preprocessing_kmer
from transformers import BertForMaskedLM, BertTokenizer
import json
import torch
from tqdm import tqdm

def parse_args(argvs):
    opts, args = getopt(argvs, "m:t:d:l:", [
            "model=",
            "test_data=",
            "device=",
            "log=",
        ])
    output = {}
    for opt, arg in opts:
        if opt in ["-m", "--model"]:
            output["model"] = arg
        elif opt in ["-t", "--test_data"]:
            output["test_data"] = arg
        elif opt in ["-d", "--device"]:
            output["device"] = arg
        elif opt in ["-l", "--log"]:
            output["log"] = arg
        else:
            print(f"Argument {opt} value {arg} not recognized.")
            sys.exit(2)
    return output


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    for key in args.keys():
        print(f"# {key} - {args[key]}")

    assert os.path.exists(args["model"]), f"Model dir not found at {args['model']}"
    assert os.path.isdir(args["model"]), f"Path {args['model']} is not directory."
    assert os.path.exists(args["test_data"]), f"Test data not found at {args['test_data']}."

    device = args["device"]

    model_config_path = os.path.join(args["model"], "model_config.json")
    model_checkpoint = os.path.join(args["model"], "model.pth")

    assert os.path.exists(model_config_path), f"Model config not found at {model_config_path}"
    assert os.path.exists(model_checkpoint), f"Model checkpoint not found at {model_checkpoint}"

    model = DNABERT_SL(
        BertForMaskedLM.from_pretrained(os.path.join("pretrained", "3-new-12w-0")).bert, # bert, 
        json.load(open(model_config_path)) # config
    )
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.to(device)

    csv_file = args["test_data"]
    tokenizer = BertTokenizer.from_pretrained(os.path.join("pretrained", "3-new-12w-0"))
    batch_size = 1
    test_dataloader = preprocessing_kmer(csv_file, tokenizer, batch_size)
    test_size = len(test_dataloader)

    logpath = args["log"]
    if os.path.exists(logpath):
        os.remove(logpath)
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logfile = open(logpath, "x")
    logfile.write("input_ids,prediction,target,accuracy\n")

    result = []
    for step, batch in tqdm(enumerate(test_dataloader), total=test_size, desc="Testing "):
        input_ids, attn_mask, token_type_ids, target_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            predictions, bert = model(input_ids, attn_mask)
            for inputs, pred, target_label in zip(input_ids, predictions, target_labels):
                vals, indices = torch.max(pred, 1)
                indices = [a for a in indices.tolist()]
                labels = [a for a in target_labels.tolist()]
                accuracy = [1 if a == b else 0 for a, b in zip(indices, labels)]
                accuracy = sum(accuracy) / predictions.shape[1] * 100
                pinputs = [a for a in inputs.tolist()]
                # pinputs = [tokenizer.convert_ids_to_tokens(a) for a in pinputs]

                # Write to log and append result to list.
                logfile.write(f"{' '.join(pinputs)},{' '.join(indices)},{' '.join(labels)},{accuracy}\n")
                result.append(
                    tuple(pinputs, indices, labels, accuracy)
                )

    logfile.close()
    

    

                



    

