import os
from sched import scheduler
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import adamw, adamax
import getopt, os, sys
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from multitask_learning import get_sequences, preprocessing, train, MTModel, PromoterHead, SpliceSiteHead, PolyAHead
from datetime import datetime

def _parse_arg(args):
    opts, arguments = getopt.getopt(args, "p:t:v:e:b:d:l:", ["pretrained=", "train_data=", "validation_data=", "epoch=", "batch_size=", "learning_rate=", "device=", "epsilon=", "warm_up=", "do_eval=", "log=", "limit_train=", "limit_val=", "loss_strategy=", "save_model_path=", "beta1=", "beta2="])
    output = {}
    
    for option, argument in opts:
        print('{} - {}'.format(option, argument))
        if option in ['-p', '--pretrained']:
            output['pretrained'] = os.path.join(argument)
        elif option in ['-t', '--train_data']:
            output['train_data'] = os.path.join(argument)
        elif option in ['-v', '--validation_data']:
            output['validation_data'] = os.path.join(argument)
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
        elif option in ['--warm_up']:
            output['warm_up'] = int(argument)
        elif option in ['--do_eval']:
            output['do_eval'] = bool(argument)
        elif option in ['--limit_train']:
            output['limit_train'] = int(argument)
        elif option in ['--limit_val']:
            output['limit_val'] = int(argument)
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
    validation_path = os.path.join(arguments['validation_data'])
    pretrained_path = os.path.join(arguments['pretrained'])
    epoch_size = arguments['epoch'] or 1
    batch_size = arguments['batch_size'] or 2000
    device = arguments['device'] if arguments.has_key('device') else 'cpu'
    log = os.path.join(arguments['log']) if arguments.has_key('log') else os.path.join('logs', "log_{}.txt".format(now))
    learning_rate = float(arguments['learning_rate']) if arguments.has_key('learning_rate') else 4e-4
    epsilon = float(arguments['epsilon']) if arguments.has_key('epsilon') else 1e-6
    beta1 = float(arguments['beta1']) if arguments.has_key('beta1') else 0.9
    beta2 = float(arguments['beta2']) if arguments.has_key('beta2') else 0.98
    do_eval = bool(arguments['do_eval']) if arguments.has_key('do_eval') else False
    warmup = int(arguments['warm_up']) if arguments.has_key('warm_up') else 0
    limit_train = int(arguments['limit_train']) if arguments.has_key('limit_train') else 0
    limit_valid = int(arguments['limit_val']) if arguments.has_key('limit_val') else 0
    loss_strategy = arguments['loss_strategy'] if arguments.has_key('loss_strategy') else 'sum' # Either `sum` or `average`
    save_model_path = arguments['save_model_path'] if arguments.has_key('save_model_path') else None

    """
    Initialize tokenizer using BertTokenizer with pretrained weights from DNABERT (Ji et. al., 2021).
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    train_seq, train_label_prom, train_label_ss, train_label_polya = get_sequences(train_path, n_sample=limit_train)
    validation_seq, val_label_prom, val_label_ss, val_label_polya = get_sequences(validation_path, n_sample=limit_valid)

    """
    Create dataloader.
    """
    BATCH_SIZE = batch_size
    EPOCH_SIZE = epoch_size

    train_label_prom = torch.tensor(train_label_prom, device=device)
    train_label_ss = torch.tensor(train_label_ss, device=device)
    train_label_polya = torch.tensor(train_label_polya, device=device)

    train_inputs_ids, train_masks = preprocessing(train_seq, tokenizer)
    train_data = TensorDataset(train_inputs_ids, train_masks, train_label_prom, train_label_ss, train_label_polya)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_label_prom = torch.tensor(val_label_prom, device=device)
    val_label_ss = torch.tensor(val_label_ss, device=device)
    val_label_polya = torch.tensor(val_label_polya, device=device)

    val_input_ids, val_masks = preprocessing(validation_seq, tokenizer)
    val_data = TensorDataset(val_input_ids, val_masks, val_label_prom, val_label_ss, val_label_polya)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    print('# of training data: {}'.format(len(train_seq)))  
    print('# of validation data: {}'.format(len(validation_seq)))

    """
    Initialize and train-validate model.
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
    optimizer = adamw(model.parameters(), lr=learning_rate, eps=epsilon)
    training_steps = len(train_dataloader) * EPOCH_SIZE
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=training_steps)

    trained_model = train(train_dataloader, model, loss_fn, optimizer, scheduler, BATCH_SIZE, EPOCH_SIZE, log, device, True, val_dataloader)

    """
    Save model.
    """

    bert_model_path = os.path.join('result', 'cpu', 'bert', now)
    whole_model_path = os.path.join('result', 'cpu', 'whole', now)
    if device != 'cpu':
        bert_model_path = os.path.join('result', 'gpu', 'bert', now)
        whole_model_path = os.path.join('result', 'gpu', 'whole', now)
    os.makedirs(bert_model_path, exist_ok=True)
    print("Saving BERT layer and whole model.")
    trained_model.shared_layer.save_pretrained(bert_model_path)
    torch.save(trained_model.state_dict(), whole_model_path)


