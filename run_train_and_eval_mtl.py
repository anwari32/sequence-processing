from sched import scheduler
from click import argument
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import getopt, os, sys
from transformers import AdamW, BertTokenizer, BertModel, get_linear_schedule_with_warmup
from multitask_learning import get_sequences, preprocessing, train, MTModel, PromoterHead, SpliceSiteHead, PolyAHead



def _parse_arg(args):
    opts, arguments = getopt.getopt(args, "p:t:v:e:b:d:l:", ["pretrained=", "train_data=", "validation_data=", "epoch=", "batch_size=", "learning_rate=", "device=", "epsilon=", "warm_up=", "do_eval=", "log=", "limit_train=", "limit_val="])
    output = {}
    for option, argument in opts:
        print('{} - {}'.format(option, argument))
        if option in ['-p', '--pretrained']:
            output['pretrained'] = os.path.join(argument)
        elif option in ['-t', '--train_data']:
            output['train_data'] = os.path.join(argument)
        elif option in ['-v', '--validation_data']:
            output['validation_data'] = os.path.join(argument)
        elif option in ['-e', '--epoch']:
            output['epoch'] = int(argument)
        elif option in ['-b', '--batch_size']:
            output['batch_size'] = int(argument)
        elif option in ['--learning_rate']:
            output['learning_rate'] = float(argument)
        elif option in ['-d', '--device']:
            output['device'] = argument
        elif option in ['-l', '--log']:
            output['log'] = argument
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
        else:
            print("Argument {} not recognized.".format(option))
            sys.exit(2)
    return output

if __name__ == "__main__":
    arguments = _parse_arg(sys.argv[1:])
    train_path = os.path.join(arguments['train_data'])
    validation_path = os.path.join(arguments['validation_data'])
    pretrained_path = os.path.join(arguments['pretrained'])
    epoch_size = arguments['epoch'] or 1
    batch_size = arguments['batch_size'] or 1
    device = arguments['device'] or 'cpu'
    log = os.path.join(arguments['log'])
    learning_rate = arguments['learning_rate'] or 5e-5
    epsilon = arguments['epsilon'] or 1e-8
    do_eval = bool(arguments['do_eval']) or False
    warmup = int(arguments['warm_up']) or 0
    limit_train = int(arguments['limit_train']) or 0
    limit_valid = int(arguments['limit_val']) or 0

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
    print('# of training data: {}'.format(len(validation_seq)))

    """
    Initialize model.
    """
    prom_head = PromoterHead(device)
    ss_head = SpliceSiteHead(device)
    polya_head = PolyAHead(device)
    bert_layer = BertModel.from_pretrained(pretrained_path)
    model = MTModel(bert_layer, prom_head, ss_head, polya_head).to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    training_steps = len(train_dataloader) * EPOCH_SIZE
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=training_steps)

    train(train_dataloader, model, loss_fn, optimizer, scheduler, BATCH_SIZE, EPOCH_SIZE, log, device, True, val_dataloader)
