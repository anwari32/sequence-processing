from getopt import getopt
import sys
import os

from transformers import get_linear_schedule_with_warmup
from sequential_labelling import preprocessing, train_using_genes, init_adamw_optimizer, init_seqlab_model
import torch




def _parse_argv(argvs):
    result = {}
    opts, arguments = getopt(argvs, "p:t:e:b:d:s:", [
        "pretrained="
        "train_dir=",
        "num_epoch=",
        "batch_size=",
        "device=",
        "save_path=",
        "log=",
        "grad_accumulation_steps=",
        "learning_rate=",
        "epsilon=",
        "beta1=",
        "beta2="
    ])
    for opt, arg in opts:
        if opt in ["-p", "--pretrained"]:
            result['pretrained'] = arg
        elif opt in ["-t", "--train_dir"]:
            result['train_dir'] = arg
        elif opt in ["-e", "--num_epoch"]:
            result['num_epochs'] = int(arg)
        elif opt in ["-b", "--batch_size"]:
            result['batch_size'] = int(arg)
        elif opt in ["-d", "--device"]:
            result['device'] = arg
        elif opt in ["-s", "--save_path="]:
            result['save_path'] = arg
        elif opt in ["--grad_accumulation_steps="]:
            result['grad_accumulation_step'] = int(arg)
        elif opt in ["--log"]:
            result["log"] = arg
        elif opt in ["--learning_rate"]:
            result["learning_rate"] = float(arg)
        elif opt in ["--epsilon"]:
            result["epsilon"] = float(arg)
        elif opt in ["--beta1"]:
            result["beta1"] = float(arg)
        elif opt in ["--beta2"]:
            result["beta2"] = float(arg)
        else:
            print(f"Argument {opt} not recognized.")
            sys.exit(2)
    return result


#   TODO:
#   Implements `learning_rate`, `beta1`, `beta2`, and `weight_decay` on AdamW optimizer.
#   Implements `loss_function`.
#   Implements  `scheduler`
if __name__ == "__main__":
    print("Training Seq2Seq model with Genes.")
    args = _parse_argv(sys.argv[1:])

    model = init_seqlab_model(args['pretrained'])
    optimizer = init_adamw_optimizer(model.parameters())

    train_genes = [os.path.join(args['train_dir'], gene) for gene in os.listdir(args['train_dir'])]
    loss_function = torch.nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(optimizer)
    model = train_using_genes(
        model, 
        optimizer, 
        scheduler, 
        train_genes, 
        loss_function, 
        num_epochs=args["num_epochs"], 
        batch_size=args["batch_size"], 
        grad_accumulation_step=args["grad_accumulation_steps"],
        device=args["device"])

