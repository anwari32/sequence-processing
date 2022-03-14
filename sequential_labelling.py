from concurrent.futures import process
from genericpath import exists
from msilib import sequence
import torch
from torch import tensor
from torch.nn import NLLLoss
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, BertForMaskedLM, get_linear_schedule_with_warmup
import os
import pandas as pd
from tqdm import tqdm
import sys
from utils.utils import save_model_state_dict, load_model_state_dict, load_checkpoint, save_checkpoint
from data_preparation import str_kmer

Labels = [
    '...',
    '..E',
    '.E.',
    'E..',
    '.EE',
    'EE.',
    'E.E',
    'EEE',
]


def _create_one_hot_encoding(index, n_classes):
    return [1 if i == index else 0 for i in range(n_classes)]

Label_Begin = '[BEGIN]'
Label_End = '[END]'
Label_Dictionary = {
    '[BEGIN]': 0, #_create_one_hot_encoding(0, 10),
    '...': 1, #_create_one_hot_encoding(1, 10),
    '..E': 2, #_create_one_hot_encoding(2, 10),
    '.E.': 3, #_create_one_hot_encoding(3, 10),
    'E..': 4, #_create_one_hot_encoding(4, 10),
    '.EE': 5, #_create_one_hot_encoding(5, 10),
    'EE.': 6, #_create_one_hot_encoding(6, 10),
    'E.E': 7, #_create_one_hot_encoding(7, 10),
    'EEE': 8, #_create_one_hot_encoding(8, 10),
    '[END]': 9, #_create_one_hot_encoding(9, 10)
}

from models.seq2seq import DNABERTSeq2Seq

def restore_model_state_dict(pretrained_path, model_state_dict_save_path):
    """
    Reinitialize saved model by initializing model and then load the saved state dictionary.
    @param  pretrained_path (string): path to pretrained model as model's initial state.
    @param  model_state_dict_save_path (string): path to saved model states.
    @return (model): model with restored parameters.
    """
    if not os.path.exists(model_state_dict_save_path):
        raise FileNotFoundError(model_state_dict_save_path)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)
    
    model = DNABERTSeq2Seq(pretrained_path)
    model.load_state_dict(torch.load(model_state_dict_save_path))
    return model

def get_sequential_labelling(csv_file, do_kmer=False, kmer_size=3):
    df = pd.read_csv(csv_file)
    sequences = list(df['sequence'])
    labels = list(df['label'])
    if do_kmer:
        sequences = [str_kmer(s, kmer_size) for s in sequences]
        labels = [str_kmer(s, kmer_size) for s in labels]
    return sequences, labels

def process_label(label_sequences, label_dict=Label_Dictionary):
    """
    @param  label_sequences (string): string containing label in kmers separated by spaces. i.e. 'EEE E.. ...'.
    @param  label_dict (map): object to map each kmer into number. Default is `Label_Dictionary`.
    @return (array of integer)
    """
    label = ['[BEGIN]']
    label.extend(label_sequences.strip().split(' '))
    label.extend(['[END]'])
    label_kmers = [label_dict[k] for k in label]
    return label_kmers

def process_sequence_and_label(sequence, label, tokenizer):
    encoded = tokenizer.encode_plus(text=sequence, return_attention_mask=True, return_token_type_ids=True, padding="max_length")
    input_ids = encoded.get('input_ids')
    attention_mask = encoded.get('attention_mask')
    token_type_ids = encoded.get('token_type_ids')
    label_repr = process_label(label)
    return input_ids, attention_mask, token_type_ids, label_repr

def create_dataloader(arr_input_ids, arr_attn_mask, arr_token_type_ids, arr_label_repr, batch_size):
    arr_input_ids_tensor = tensor(arr_input_ids)
    arr_attn_mask_tensor = tensor(arr_attn_mask)
    arr_token_type_ids_tensor = tensor(arr_token_type_ids)
    arr_label_repr_tensor = tensor(arr_label_repr)

    dataset = TensorDataset(arr_input_ids_tensor, arr_attn_mask_tensor, arr_token_type_ids_tensor, arr_label_repr_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def preprocessing(csv_file, tokenizer, batch_size, do_kmer=False, kmer_size=3):
    """
    @return dataloader (torch.utils.data.DataLoader): dataloader
    """
    sequences, labels = get_sequential_labelling(csv_file, do_kmer=do_kmer, kmer_size=kmer_size)
    arr_input_ids = []
    arr_attention_mask = []
    arr_token_type_ids = []
    arr_labels = []
    for seq, label in zip(sequences, labels):
        input_ids, attention_mask, token_type_ids, label_repr = process_sequence_and_label(seq, label, tokenizer)
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_labels.append(label_repr)

    tensor_dataloader = create_dataloader(arr_input_ids, arr_attention_mask, arr_token_type_ids, arr_labels, batch_size)
    return tensor_dataloader

def init_adamw_optimizer(model_parameters, learning_rate=1e-5, epsilon=1e-6, betas=(0.9, 0.98), weight_decay=0.01):
    """
    Initialize AdamW optimizer.
    @param  model_parameters:
    @param  learning_rate: Default is 1e-5 so it's small. 
            Change to 2e-4 for fine-tuning purpose (Ji et. al., 2021) or 4e-4 for pretraining (Ji et. al., 2021).
    @param  epsilon: adam epsilon, default is 1e-6 as in DNABERT pretraining (Ji et. al., 2021).
    @param  betas: a tuple consists of beta 1 and beta 2.
    @param  weight_decay: weight_decay
    @return (AdamW object)
    """
    optimizer = AdamW(model_parameters, lr=learning_rate, eps=epsilon, betas=betas, weight_decay=weight_decay)
    return optimizer

def train_iter(args):
    for epoch in args.num_epoch:
        model = train(args.model, args.optimizer, args.scheduler, args.batch_size, args.log)

def train_and_eval(model, train_dataloader, valid_dataloader, device="cpu"):
    model.to(device)
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids, attn_mask, label_prom, label_ss, label_polya = tuple(t.to(device) for t in batch)
        output = model(input_ids, attn_mask)
    return model

def train(model, optimizer, scheduler, train_dataloader, epoch_size, batch_size, log_path, save_model_path, device='cpu', remove_old_model=False, training_counter=0, resume_from_checkpoint=None, resume_from_optimizer=None):
    """
    @param  model: BERT derivatives.
    @param  optimizer: optimizer
    @param  scheduler:
    @param  train_dataloader:
    @param  validation_dataloader:
    @param  epoch_size:
    @param  batch_size:
    @param  log_path (string):
    @param  save_model_path (string): where to save model for each epoch.
    @param  device (string) | None -> 'cpu': Default value is 'cpu', can be changed into 'cuda', 'cuda:0' for first cuda-compatible device, 'cuda:1' for second device, etc.
    @return model after training.
    """
    # Make directories if directories does not exist.
    if not os.path.dirname(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # If log file exists, quit training.
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_file = open(log_path, 'x')
    log_file.write('epoch,step,loss,learning_rate\n')
    model.to(device)
    model.train()
    for i in range(epoch_size):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids, attention_mask, input_type_ids, label = tuple(t.to(device) for t in batch)
            model.zero_grad()
            pred = model(input_ids, attention_mask, input_type_ids)
            loss_batch = None
            for p, t in zip(pred, label):
                loss = model.loss_function(p, t)
                if loss_batch == None:
                    loss_batch = loss
                else:
                    loss_batch += loss
            if model.loss_strategy == "average":
                loss_batch = loss_batch / batch_size
            lr = optimizer.param_groups[0]['lr']
            log_file.write(f"{i+training_counter},{step},{loss_batch},{lr}\n")
            loss_batch.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
        #endfor batch

        # After an epoch, save model state.
        save_model_state_dict(model, save_model_path, "epoch-{}.pth".format(i+training_counter))
        save_model_state_dict(optimizer, save_model_path, "optimizer-{}.pth".format(i+training_counter))
        if remove_old_model:
            old_model_path = os.path.join(save_model_path, os.path.basename("epoch-{}.pth".format(i+training_counter-1)))

    #endfor epoch
    log_file.close()
    return model

def evaluate(model, validation_dataloader, log, batch_size=1, device='cpu'):
    model.to(device)
    model.eval()
    for step, batch in tqdm(enumerate(validation_dataloader, total=len(validation_dataloader))):
        input_ids, attention_mask, label = tuple(t.to(device) for t in batch)
        pred = model(input_ids, attention_mask)
        

def convert_pred_to_label(pred, label_dict=Label_Dictionary):
    """
    @param      pred: tensor (<seq_length>, <dim>)
    @param      label_dict: 
    @return     array: []
    """
    return []


# def train_using_gene(model, tokenizer, optimizer, scheduler, num_epoch, batch_size, train_genes, loss_function, grad_accumulation_step="1", device="cpu"):
def train_using_genes(model, tokenizer, optimizer, scheduler, train_genes, loss_function, num_epoch=1, batch_size=1, grad_accumulation_step="1", device="cpu", resume_checkpoint=None, save_path=None):
    """
    @param  model
    @param  tokenizer
    @param  optimizer
    @param  scheduler
    @param  num_epoch (int | None -> 1)
    @param  batch_size (int | None -> 1)
    @param  train_genes (list<string>) : list of gene file path.
    @param  loss_function
    @param  grad_accumulation_step (int | None -> 1)
    @param  device (str | None -> ``cpu``)
    @return ``model``
    """
    resume_training_counter = 0
    if resume_checkpoint != None:
        model, optimizer, config = load_checkpoint(resume_checkpoint, model, optimizer)
        resume_training_counter = config["epoch"]
    
    num_training_genes = len(train_genes)
    for epoch in range(num_epoch):
        epoch_loss = None
        for i in range(num_training_genes):
            
            gene = train_genes[i]
            gene_dataloader = create_dataloader(gene, batch_size, tokenizer) # Create dataloader for this gene.
            gene_loss = None # This is loss computed from single gene.
            len_dataloader = len(gene_dataloader)
            total_training_instance = len_dataloader * batch_size # How many small sequences are in training.
            for step, batch in tqdm(enumerate(gene_dataloader), total=len_dataloader):
                input_ids, attn_mask, token_type_ids, label = tuple(t.to(device) for t in batch)

                pred = model(input_ids, attn_mask, token_type_ids)
                batch_loss = loss_function(pred, label)
                gene_loss = batch_loss if gene_loss == None else gene_loss + batch_loss
            #endfor

            avg_gene_loss = gene_loss / total_training_instance
            epoch_loss = avg_gene_loss if epoch_loss == None else epoch_loss + avg_gene_loss
            avg_gene_loss.backward()

            if i % grad_accumulation_step == 0 or (i + 1) == num_training_genes:
                optimizer.step()
                scheduler.step()

        #endfor
        save_checkpoint(model, optimizer, {
            'epoch': epoch + 1 + resume_training_counter,
            'loss': epoch_loss
        }, save_path)
    #endfor

    return model