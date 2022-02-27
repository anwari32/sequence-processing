import os
from sched import scheduler
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim.adamw import AdamW
import getopt, os, sys
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from multitask_learning import train, MTModel, PromoterHead, SpliceSiteHead, PolyAHead, prepare_data
from datetime import datetime

def _parse_arg(args):
    opts, arguments = getopt.getopt(args, "p:t:e:b:d:l:", 
        ["pretrained=", 
        "train_data=", 
        "epoch=", 
        "batch_size=", 
        "device=", 
        "learning_rate=", 
        "epsilon=", 
        "beta1=", "beta2="
        "weight_decay=",
        "warm_up=", 
        "log=", 
        "limit_train=", 
        "loss_strategy=", 
        "save_model_path="]
    )
    output = {}
    
    for option, argument in opts:
        if option in ['-p', '--pretrained']:
            output['pretrained'] = os.path.join(argument)
        elif option in ['-t', '--train_data']:
            output['train_data'] = os.path.join(argument)
        elif option in ['-d', '--device']:
            output['device'] = argument
        elif option in ['-e', '--epoch']:
            output['epoch'] = int(argument)
        elif option in ['-b', '--batch_size']:
            output['batch_size'] = int(argument)
        elif option in ['--learning_rate']:
            output['learning_rate'] = float(argument)
        elif option in ['--epsilon']:
            output['epsilon'] = float(argument)
        elif option in ['beta1']:
            output['beta1'] = float(argument)
        elif option in ['beta2']:
            output['beta2'] = float(argument)
        elif option in ['--warm_up']:
            output['warm_up'] = int(argument)
        elif option in ['--limit_train']:
            output['limit_train'] = int(argument)
        elif option in ['--loss_strategy']:
            output['loss_strategy'] = argument
        elif option in ['--optimizer']:
            output['optimizer'] = argument
        elif option in ['--save_model_path']:
            output['save_model_path'] = argument
        elif option in ['-l', '--log']:
            output['log'] = argument
        else:
            print("Argument {} not recognized.".format(option))
            sys.exit(2)
    return output

if __name__ == "__main__":
    now = datetime.now().strftime('%Y-%m-%d')
    arguments = _parse_arg(sys.argv[1:])
    train_path = os.path.join(arguments['train_data'])
    pretrained_path = os.path.join(arguments['pretrained'])
    epoch_size = int(arguments['epoch']) if 'epoch' in arguments.keys() else 1
    batch_size = int(arguments['batch_size']) if 'batch_size' in arguments.keys() else 2000
    device = arguments['device'] if 'device' in arguments.keys() else 'cpu'
    log = os.path.join(arguments['log']) if 'log' in arguments.keys() else os.path.join('logs', "log_{}.txt".format(now))
    learning_rate = float(arguments['learning_rate']) if 'learning_rate' in arguments.keys() else 4e-4
    epsilon = float(arguments['epsilon']) if 'epsilon' in arguments.keys() else 1e-6
    beta1 = float(arguments['beta1']) if 'beta1' in arguments.keys() else 0.9
    beta2 = float(arguments['beta2']) if 'beta2' in arguments.keys() else 0.98
    weight_decay = float(arguments['weight_decay']) if 'weight_decay' in arguments.keys() else 0.01
    warmup = int(arguments['warm_up']) if 'warm_up' in arguments.keys() else 0
    limit_train = int(arguments['limit_train']) if 'limit_train' in arguments.keys() else 0
    loss_strategy = arguments['loss_strategy'] if 'loss_strategy' in arguments.keys() else 'sum' # Either `sum` or `average`
    save_model_path = arguments['save_model_path'] if 'save_model_path' in arguments.keys() else None

    for key in arguments.keys():
        print('{} - {}'.format(key, arguments[key]))
        
    """
    Create dataloader.
    """
    BATCH_SIZE = batch_size
    EPOCH_SIZE = epoch_size

    train_dataloader = prepare_data(train_path, pretrained_path, batch_size=BATCH_SIZE, n_sample=limit_train)

    print('# of training data: {}'.format(len(train_dataloader)))

    """
    Initialize model, optimizer, and scheduler.
    """
    prom_head = PromoterHead(device)
    ss_head = SpliceSiteHead(device)
    polya_head = PolyAHead(device)
    bert_layer = BertModel.from_pretrained(pretrained_path)
    model = MTModel(bert_layer, prom_head, ss_head, polya_head).to(device)
    loss_fn = {
        'prom':BCELoss(), 
        'ss': CrossEntropyLoss(), 
        'polya': CrossEntropyLoss()
    }
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon, betas=(beta1, beta2), weight_decay=weight_decay)
    training_steps = len(train_dataloader) * EPOCH_SIZE
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=training_steps)

    trained_model = train(train_dataloader, model, loss_fn, optimizer, scheduler, BATCH_SIZE, EPOCH_SIZE, log, device, loss_strategy=loss_strategy, save_model_path=save_model_path)
