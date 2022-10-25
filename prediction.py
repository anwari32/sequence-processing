from multiprocessing.sharedctypes import Value
import sys
import os
from getopt import getopt
from models.seqlab import DNABERT_SL
from utils.seqlab import NUM_LABELS, Index_Dictionary, preprocessing_kmer
from transformers import BertForMaskedLM, BertTokenizer
import json
import torch
from tqdm import tqdm
from utils.metrics import Metrics
import wandb

def parse_args(argvs):
    opts, args = getopt(argvs, "w:c:t:d:l:", [
            "work-dir=",
            "model-config=",
            "test-data=",
            "device=",
            "log=",
        ])
    output = {}
    for opt, arg in opts:
        if opt in ["-c", "--model-config"]:
            output["model-config"] = arg
        elif opt in ["-t", "--test-data"]:
            output["test_data"] = arg
        elif opt in ["-d", "--device"]:
            output["device"] = arg
        elif opt in ["-l", "--log"]:
            output["log"] = arg
        elif opt in ["-w", "--work-dir"]:
            output["work_dir"]
        else:
            print(f"Argument {opt} value {arg} not recognized.")
            sys.exit(2)
    return output


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    for key in args.keys():
        print(f"# {key} - {args[key]}")

    device = args.get("device", "cpu") # specify device or use cpu otherwise.

    model_config_path = args.get("model-config", False)
    model_checkpoint = args.get("model-checkpoint", False)
    test_config = args.get("test-config")
    test_file = test_config.gt("test_data", False)

    if not model_config_path:
        raise ValueError("model config not specified.")
    if not os.path.exists(model_config_path):
        raise ValueError(f"model config not exists at {model_config_path}")

    if not model_checkpoint:
        raise ValueError("model checkpoint not specified.")
    if not os.path.exists(model_checkpoint):
        raise ValueError(f"model checkpoint not exists at {model_checkpoint}")

    if not test_file:
        raise ValueError("test not specified.")
    if not os.path.exists(test_file):
        raise ValueError(f"test file not exists at {test_file}")

    bert_for_masked_lm = BertForMaskedLM.from_pretrained(os.path.join("pretrained", "3-new-12w-0"))
    model = DNABERT_SL(
        bert_for_masked_lm.bert, # bert, 
        json.load(open(model_config_path)) # config
    )

    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model"))
    model.eval()
    model.to(device)


    tokenizer = BertTokenizer.from_pretrained(os.path.join("pretrained", "3-new-12w-0"))
    batch_size = 1
    test_dataloader = preprocessing_kmer(test_file, tokenizer, batch_size)
    test_size = len(test_dataloader)

    logpath = args["log"]
    if os.path.exists(logpath):
        os.remove(logpath)
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logfile = open(logpath, "x")
    logfile.write("input_ids,prediction,target,accuracy\n")

    # initialize wandb.
    wandb.init(
        project="prediction",
        entity="anwari32"
    )
    wandb.define_metric("prediction_step")
    wandb.define_metric("prediction/*", step_metric="prediction_step")

    result = []
    prediction_step = 0
    for step, batch in tqdm(enumerate(test_dataloader), total=test_size, desc="Testing"):
        input_ids, attn_mask, token_type_ids, target_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            predictions, bert_output = model(input_ids, attn_mask)
            for inputs, pred, target_label in zip(input_ids, predictions, target_labels):
                vals, pred_ids = torch.max(pred, 1)
                actual_input_ids = input_ids[1:] # remove CLS token
                actual_input_ids = [t for t in actual_input_ids if t > 0]
                actual_pred_ids = pred_ids[1:] # remove CLS prediction
                actual_pred_ids = actual_pred_ids[0:len(actual_input_ids)]
                target_ids = target_label[1:]
                target_ids = target_label[0:len(actual_input_ids)]

                metrics = Metrics(actual_pred_ids, target_ids)
                for label_idx in range(NUM_LABELS):
                    wandb.log({
                        f"precision-{Index_Dictionary[label_idx]}": metrics.precision(label_idx),
                        f"recall-{Index_Dictionary[label_idx]}": metrics.recall(label_idx),
                        
                        "prediction_step": prediction_step
                    })

                prediction_step += 1
                


    logfile.close()
    

    

                



    

