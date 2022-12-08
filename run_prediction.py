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

if __name__ == "__main__":
    # args = parse_args(sys.argv[1:])
    # for key in args.keys():
    #     print(f"# {key} - {args[key]}")

    # use static input first.
    args = {
        "device": "cpu",
        "model-config": os.path.join("models", "config", "seqlab", "base.lin1.json"),
        "model-checkpoint": os.path.join("run", "sso01-adamw-lr5e-5-base.lin1-2w1boplw", "latest", "checkpoint.pth"),
        "test-config": os.path.join("training", "config", "seqlab", "ss-only.01.lr5e-5.json"),
        "test-file": os.path.join("workspace", "seqlab-latest", "gene_index.01_test_ss_all_pos.csv"),
        "log": os.path.join("prediction", "2w1boplw", "prediction_log.csv")
    }

    device = args.get("device", "cpu") # specify device or use cpu otherwise.

    model_config_path = args.get("model-config", False)
    model_checkpoint = args.get("model-checkpoint", False)
    test_config_path = args.get("test-config")
    # test_config = json.load(open(test_config_path, "r"))
    # test_file = test_config.get("test_data", False)
    test_file = args.get("test-file", None)

    if not model_config_path:
        raise ValueError("model config not specified.")
    if not os.path.exists(model_config_path):
        raise ValueError(f"model config not exists at {model_config_path}")
    print(f"using model config at {model_config_path}")
        
    if not model_checkpoint:
        raise ValueError("model checkpoint not specified.")
    if not os.path.exists(model_checkpoint):
        raise ValueError(f"model checkpoint not exists at {model_checkpoint}")
    print(f"found model checkpoint at {model_checkpoint}")

    if not test_file:
        raise ValueError("test not specified.")
    if not os.path.exists(test_file):
        raise ValueError(f"test file not exists at {test_file}")
    print(f"found test data at {test_file}")

    bert_for_masked_lm = BertForMaskedLM.from_pretrained(os.path.join("pretrained", "3-new-12w-0"))
    model = DNABERT_SL(
        bert_for_masked_lm.bert, # bert, 
        json.load(open(model_config_path, "r")) # config
    )

    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model"))
    model.eval()
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(os.path.join("pretrained", "3-new-12w-0"))
    batch_size = 1
    test_dataloader = preprocessing_kmer(test_file, tokenizer, batch_size)
    test_size = len(test_dataloader)

    # initialize wandb.
    run = wandb.init(
        project="prediction",
        entity="anwari32"
    )
    wandb.define_metric("prediction_step")
    wandb.define_metric("prediction/*", step_metric="prediction_step")

    logpath = args.get("log", "prediction")
    logpath = os.path.join(logpath, f"{run.id}.csv")
    if os.path.exists(logpath):
        os.remove(logpath)
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logfile = open(logpath, "x")
    logfile.write("step,input_ids,prediction,target\n")


    result = []
    prediction_step = 0
    device = "cuda:0"
    model.to(device)
    for step, batch in tqdm(enumerate(test_dataloader), total=test_size, desc="Testing"):
        input_ids, attn_mask, token_type_ids, target_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            predictions, bert_output = model(input_ids, attn_mask)
            for inputs, pred, target_label in zip(input_ids, predictions, target_labels):
                vals, pred_ids = torch.max(pred, 1)
                input_ids_list = inputs.tolist()
                pred_ids_list = pred_ids.tolist()
                target_list = target_label.tolist()
                
                # log to local first.
                input_ids_str = [str(a) for a in input_ids_list]
                input_ids_str = " ".join(input_ids_str)
                pred_ids_str = [str(a) for a in pred_ids_list]
                pred_ids_str = " ".join(pred_ids_str)
                target_ids_str = [str(a) for a in target_list]
                target_ids_str = " ".join(target_ids_str)

                logfile.write(f"{prediction_step},{input_ids_str},{pred_ids_str},{target_ids_str}\n")

                actual_input_ids = input_ids_list[1:] # remove CLS token
                actual_input_ids = [t for t in actual_input_ids if t > 0] # token id 0 is padding.

                actual_target_ids = target_list[1:] # remove CLS token
                actual_target_ids = [a for a in actual_target_ids if a >= 0] # label id < 0 is special tokens.

                actual_pred_ids = pred_ids_list[1:] # remove CLS prediction
                actual_pred_ids = actual_pred_ids[0:len(actual_target_ids)]

                metrics = Metrics(actual_pred_ids, actual_target_ids)
                metrics.calculate()
                for label_idx in range(NUM_LABELS):
                    wandb.log({
                        f"prediction/precision-{Index_Dictionary[label_idx]}": metrics.precision(label_idx),
                        f"prediction/recall-{Index_Dictionary[label_idx]}": metrics.recall(label_idx),
                        f"prediction/f1_score-{Index_Dictionary[label_idx]}": metrics.f1_score(label_idx),
                        "prediction_step": prediction_step
                    })

                prediction_step += 1
                
    logfile.close()
    run.finish()