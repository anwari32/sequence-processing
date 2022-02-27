from getopt import getopt
import sys
import os
from multitask_learning import evaluate, prepare_data, restore_model
from torch.nn import BCELoss, CrossEntropyLoss
from data_dir import pretrained_3kmer_dir

def _parse_arg(args):
    opts, arguments = getopt(args, "p:e:b:d:l:s", 
        ["pretrained=", 
        "eval_data=", 
        "batch_size=", 
        "device=", 
        "log=", 
        "limit_validation=", 
        "loss_strategy="]
    )
    output = {}
    
    for option, argument in opts:
        if option in ['-p', '--pretrained']:
            output['pretrained'] = os.path.join(argument)
        elif option in ['-t', '--eval_data']:
            output['validation_data'] = os.path.join(argument)
        elif option in ['-d', '--device']:
            output['device'] = argument
        elif option in ['-l', '--log']:
            output['log'] = argument
        elif option in ['-b', '--batch_size']:
            output['batch_size'] = int(argument)
        elif option in ['--limit_validation']:
            output['limit_validation'] = int(argument)
        elif option in ['--loss_strategy']:
            output['loss_strategy'] = argument
        else:
            print("Argument {} not recognized.".format(option))
            sys.exit(2)
    return output

if __name__=="__main__":
    argv = sys.argv[1:]
    arguments = _parse_arg(argv)

    if 'validation_data' not in arguments.keys():
        print("No validation data is available. Please provide one.")
        sys.exit(2)

    if 'pretrained' not in arguments.keys():
        print("Location pretrained model or model that has been trained before is not available. Please provide one.")

    prepared_model_path = os.path.join(arguments['pretrained'])
    validation_path = os.path.join(arguments['validation_data'])
    batch_size = int(arguments['batch_size']) if 'batch_size' in arguments.keys() else 1
    limit_validation = int(arguments['limit_validation']) if 'limt_validation' in arguments.keys() else 0
    device = arguments['device'] if 'device' in arguments.keys() else "cpu"
    log = os.path.join(arguments['log']) if 'log' in arguments.keys() else None

    val_dataloader = prepare_data(validation_path, pretrained_3kmer_dir)
    model = restore_model(prepared_model_path)
    loss_functions = {
        'prom': BCELoss(),
        'ss': CrossEntropyLoss(),
        'polya': CrossEntropyLoss()
    }

    result = evaluate(val_dataloader, model, loss_functions, log, device=device)
    titles = ['avg prom acc', 'avg polya acc', 'avg prom loss', 'avg ss loss', 'avg polya loss']
    for a, b in zip(titles, result):
        print(a, b)