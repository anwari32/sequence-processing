from getopt import getopt
from mimetypes import init
from torch.optim.adamw import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from sequential_labelling import DNABERTSeq2Seq, preprocessing, train
import sys
import os
from datetime import datetime
import torch

from utils.utils import load_model_state_dict

def _parse_arg(argv):
    result = {}
    opts, args = getopt(argv, "t:p:", [
        "train_data=",
        "pretrained=",
        "num_epochs=",
        "batch_size=",
        "device=",
        "log=",
        "learning_rate=",
        "epsilon=",
        "beta1=",
        "beta2=",
        "weight_decay=",
        "warmup=",
        "loss_strategy=",
        "save_model_path=",
        "remove_old_model=",
        "resume_from_checkpoint=",
        "resume_from_optimizer=",
        "training_counter="
    ])
    for opt, arg in opts:
        if opt in ["--train_data", "-t"]:
            result["train_data"] = arg
        elif opt in ["--pretrained", "-p"]:
            result["pretrained"] = arg
        elif opt in ["--device"]:
            result["device"] = arg
        elif opt in ["--num_epochs"]:
            result["num_epochs"] = int(arg)
        elif opt in ["--batch_size"]:
            result["batch_size"] = int(arg)
        else:
            print(f"Argument {opt} not recognized.")
            sys.exit(2)
    return result

if __name__ == "__main__":
    print("Training Seq2Seq Model")
    
    now = datetime.now().strftime('%Y-%m-%d')
    arguments = _parse_arg(sys.argv[1:])
    for k in arguments.keys():
        print(k, arguments[k])

    train_path = os.path.join(arguments['train_data'])
    pretrained_path = os.path.join(arguments['pretrained'])
    epoch_size = int(arguments['num_epochs']) if 'num_epochs' in arguments.keys() else 1
    batch_size = int(arguments['batch_size']) if 'batch_size' in arguments.keys() else 1
    device = arguments['device'] if 'device' in arguments.keys() else 'cpu'
    log = os.path.join(arguments['log']) if 'log' in arguments.keys() else os.path.join('logs', "log_{}.csv".format(now))
    learning_rate = float(arguments['learning_rate']) if 'learning_rate' in arguments.keys() else 4e-4
    epsilon = float(arguments['epsilon']) if 'epsilon' in arguments.keys() else 1e-6
    beta1 = float(arguments['beta1']) if 'beta1' in arguments.keys() else 0.9
    beta2 = float(arguments['beta2']) if 'beta2' in arguments.keys() else 0.98
    weight_decay = float(arguments['weight_decay']) if 'weight_decay' in arguments.keys() else 0.01
    warmup = int(arguments['warm_up']) if 'warm_up' in arguments.keys() else 0
    loss_strategy = arguments['loss_strategy'] if 'loss_strategy' in arguments.keys() else 'sum' # Either `sum` or `average`
    save_model_path = arguments['save_model_path'] if 'save_model_path' in arguments.keys() else os.path.join("result", "{}".format(now))
    remove_old_model = arguments['remove_old_model'] if 'remove_old_model' in arguments.keys() else False
    resume_from_checkpoint = arguments['resume_from_checkpoint'] if 'resume_from_checkpoint' in arguments.keys() else None
    resume_from_optimizer = arguments['resume_from_optimizer'] if 'resume_from_optimizer' in arguments.keys() else None
    training_counter = arguments['training_counter'] if 'training_counter' in arguments.keys() else 0

    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    train_dataloader = preprocessing(train_path, tokenizer, batch_size, do_kmer=True)
    model = DNABERTSeq2Seq(pretrained_path)
    if resume_from_checkpoint != None:
        model = load_model_state_dict(model, resume_from_checkpoint)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon, betas=(beta1, beta2), weight_decay=weight_decay)
    if resume_from_optimizer != None:
        optimizer = load_model_state_dict(optimizer, resume_from_optimizer)
    training_steps = len(train_dataloader) * epoch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=training_steps)

    trained_model = train(model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        epoch_size, 
        batch_size, 
        log_path=log, 
        save_path=save_model_path, 
        device=device,
        training_counter=training_counter,
        resume_from_checkpoint=resume_from_checkpoint,
        resume_from_optimizer=resume_from_optimizer
    )