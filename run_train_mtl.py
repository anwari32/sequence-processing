from json import load
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
from utils.utils import load_model_state_dict

def _parse_arg(args):
    opts, arguments = getopt.getopt(args, "p:t:e:b:d:l:", 
        ["pretrained=", 
        "train_data=", 
        "num_epochs=", 
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
        "save_model_path=",
        "remove_old_model=",
        "grad_accumulation_steps=",
        "resume_from_checkpoint=",
        "resume_from_optimizer=",
        "training_counter="]
    )
    output = {}
    
    for option, argument in opts:
        if option in ['-p', '--pretrained']:
            output['pretrained'] = os.path.join(argument)
        elif option in ['-t', '--train_data']:
            output['train_data'] = os.path.join(argument)
        elif option in ['-d', '--device']:
            output['device'] = argument
        elif option in ['-e', '--num_epochs']:
            output['num_epochs'] = int(argument)
        elif option in ['-b', '--batch_size']:
            output['batch_size'] = int(argument)
        elif option in ['--learning_rate']:
            output['learning_rate'] = float(argument)
        elif option in ['--epsilon']:
            output['epsilon'] = float(argument)
        elif option in ['--beta1']:
            output['beta1'] = float(argument)
        elif option in ['--beta2']:
            output['beta2'] = float(argument)
        elif option in ['--weight_decay']:
            output['weight_decay'] = float(argument)
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
        elif option in ['--grad_accumulation_steps']:
            output['grad_accumulation_steps'] = int(argument)
        elif option in ['-l', '--log']:
            output['log'] = argument
        elif option in ['--remove_old_model']:
            output['remove_old_model'] = bool(argument)
        elif option in ['--resume_from_checkpoint']:
            output['resume_from_checkpoint'] = argument
        elif option in ['--resume_from_optimizer']:
            output['resume_from_optimizer'] = argument
        elif option in ['--training_counter']:
            output['training_counter'] = int(argument)
        else:
            print("Argument {} not recognized.".format(option))
            sys.exit(2)
    return output

def _format_logname(train_file, num_epoch, batch_size, loss_strategy, grad_accumulation_steps, date_str=None):
    if date_str == None:
        date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return "log_{}_{}_e{}_b{}_{}_g{}.csv".format(date_str, os.path.basename(train_file), num_epoch, batch_size, loss_strategy, grad_accumulation_steps)

def _format_foldername(train_file, num_epoch, batch_size, loss_strategy, grad_accumulation_steps, date_str=None):
    if date_str == None:
        date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return "model_{}_{}_e{}_b{}_{}_g{}".format(date_str, os.path.basename(train_file), num_epoch, batch_size, loss_strategy, grad_accumulation_steps)

if __name__ == "__main__":
    now = datetime.now().strftime('%Y-%m-%d')
    arguments = _parse_arg(sys.argv[1:])
    parameters = {}
    parameters['train_data'] = train_path = os.path.join(arguments['train_data']) if 'train_data' in arguments.keys() else None
    parameters['pretrained'] = pretrained_path = os.path.join(arguments['pretrained']) if 'pretrained' in arguments.keys() else None
    parameters['epoch'] = epoch_size = int(arguments['epoch']) if 'epoch' in arguments.keys() else 1
    parameters['batch_size'] = batch_size = int(arguments['batch_size']) if 'batch_size' in arguments.keys() else 2000
    parameters['device'] = device = arguments['device'] if 'device' in arguments.keys() else 'cpu'
    parameters['learning_rate'] = learning_rate = float(arguments['learning_rate']) if 'learning_rate' in arguments.keys() else 4e-4
    parameters['epsilon'] = epsilon = float(arguments['epsilon']) if 'epsilon' in arguments.keys() else 1e-6
    parameters['beta1'] = beta1 = float(arguments['beta1']) if 'beta1' in arguments.keys() else 0.9
    parameters['beta2'] = beta2 = float(arguments['beta2']) if 'beta2' in arguments.keys() else 0.98
    parameters['weight_decay'] = weight_decay = float(arguments['weight_decay']) if 'weight_decay' in arguments.keys() else 0.01
    parameters['warmup'] = warmup = int(arguments['warm_up']) if 'warm_up' in arguments.keys() else 0
    parameters['limit_train'] = limit_train = int(arguments['limit_train']) if 'limit_train' in arguments.keys() else 0
    parameters['loss_strategy'] = loss_strategy = arguments['loss_strategy'] if 'loss_strategy' in arguments.keys() else 'sum' # Either `sum` or `average`
    parameters['grad_accumulation_steps'] = grad_accumulation_steps = arguments['grad_accumulation_steps'] if 'grad_accumulation_steps' in arguments.keys() else 1
    parameters['remove_old_model'] = remove_old_model = arguments['remove_old_model'] if 'remove_old_model' in arguments.keys() else False
    parameters['resume_from_checkpoint'] = resume_from_checkpoint = arguments['resume_from_checkpoint'] if 'resume_from_checkpoint' in arguments.keys() else None
    parameters['resume_from_optimizer'] = resume_from_optimizer = arguments['resume_from_optimizer'] if 'resume_from_optimizer' in arguments.keys() else None
    parameters['training_counter'] = training_counter = arguments['training_counter'] if 'training_counter' in arguments.keys() else 0
    parameters['save_model_path'] = save_model_path = arguments['save_model_path'] if 'save_model_path' in arguments.keys() else os.path.join("result", now, _format_foldername(train_path, epoch_size, batch_size, loss_strategy, grad_accumulation_steps))
    parameters['log'] = log = os.path.join(arguments['log']) if 'log' in arguments.keys() else os.path.join("logs", now, _format_logname(train_path, epoch_size, batch_size, loss_strategy, grad_accumulation_steps))

    for key in parameters.keys():
        print('{} - {}'.format(key, parameters[key]))
        
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
    model = MTModel(bert_layer, prom_head, ss_head, polya_head)
    if resume_from_checkpoint != None:
        print(f"Loading existing model to continue training <{resume_from_checkpoint}>")
        model = load_model_state_dict(model, resume_from_checkpoint)
    model.to(device)
    loss_fn = {
        'prom':BCELoss(), 
        'ss': CrossEntropyLoss(), 
        'polya': CrossEntropyLoss()
    }
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon, betas=(beta1, beta2), weight_decay=weight_decay)
    if resume_from_optimizer != None:
        print(f"Loading existing optimizer to continue training <{resume_from_optimizer}>")
        optimizer = load_model_state_dict(optimizer, resume_from_optimizer)
    training_steps = len(train_dataloader) * EPOCH_SIZE
    optim_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=training_steps)

    trained_model = train(
        train_dataloader, 
        model, 
        loss_fn, 
        optimizer, 
        optim_scheduler, 
        BATCH_SIZE, 
        EPOCH_SIZE, 
        log, 
        device, 
        loss_strategy=loss_strategy, 
        save_model_path=save_model_path, 
        remove_old_model=remove_old_model,
        training_counter=training_counter,
        grad_accumulation_steps=grad_accumulation_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        resume_from_optimizer=resume_from_optimizer)
