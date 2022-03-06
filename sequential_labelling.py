from concurrent.futures import process
from genericpath import exists
from msilib import sequence
import torch
from torch import nn, tensor
from torch.nn import NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, BertForMaskedLM, get_linear_schedule_with_warmup
import os
import pandas as pd
from tqdm import tqdm
import sys
from utils import save_model_state_dict

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


Label_Begin = '[BEGIN]'
Label_End = '[END]'
Label_Dictionary = {
    '[BEGIN]': 0,
    '...': 1,
    '..E': 2,
    '.E.': 3,
    'E..': 4,
    '.EE': 5,
    'EE.': 6,
    'E.E': 7,
    'EEE': 8,
    '[END]': 9
}

class Seq2SeqHead(nn.Module):
    def __init__(self, dims):
        super().__init__()
        dims_ins_outs = [dims[i:i+2] for i in range(len(dims)-2+1)]
        self.hidden_layers = [nn.Linear(d[0], d[1]) for d in dims_ins_outs]
        self.stack = nn.Sequential()
        self.activation = nn.LogSoftmax(dim=1)
        for i in range(0, len(self.hidden_layers)):
            linear_layer = self.hidden_layers[i]
            self.stack.add_module("hidden-{}".format(i+1), linear_layer)
            self.stack.add_module("relu-{}".format(i+1), nn.ReLU())
        self.stack.add_module("dropout-1", nn.Dropout(0.1))
    
    def forward(self, input):
        x = self.stack(input)
        x = self.activation(x)
        return x

class DNABERTSeq2Seq(nn.Module):
    """
    Core architecture of sequential labelling.
    """
    def __init__(self, bert_pretrained_path, seq2seq_dims=[768, 512, 512, 10], loss_strategy="sum", device='cpu'):
        """
        This model uses BERT as its feature extraction layer.
        This BERT layer is initiated from pretrained model which is located at `bert_pretrained_path`.
        @param  bert_pretrained_path (string): Path to DNABERT pretrained.
        @param  seq2seq_dims:
        @param  loss_strategy (string) | None -> "sum"
        @param  device (string): Default is 'cpu' but you can put 'cuda' if your machine supports cuda.
        @return (DNASeqLabelling): Object of this class.
        """
        if not os.path.exists(bert_pretrained_path):
            raise FileNotFoundError(bert_pretrained_path)
        if not os.path.isdir(bert_pretrained_path):
            raise IsADirectoryError(bert_pretrained_path)

        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_pretrained_path).bert
        self.seq2seq_head = Seq2SeqHead(seq2seq_dims)
        self.loss_function = nn.NLLLoss()
        self.loss_strategy = loss_strategy
        self.activation = nn.Softmax(dim=2)

    def forward(self, input_ids, attention_masks):
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        output = output[0]
        output = self.seq2seq_head(output)
        output = self.activation(output)
        return output

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

def get_sequential_labelling(csv_file):
    df = pd.read_csv(csv_file)
    sequences = list(df['sequence'])
    labels = list(df['label'])
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
    encoded = tokenizer.encode_plus(text=sequence, return_attention_mask=True, padding="max_length")
    input_ids = encoded.get('input_ids')
    attention_mask = encoded.get('attention_mask')
    label_repr = process_label(label)
    return input_ids, attention_mask, label_repr

def create_dataloader(arr_input_ids, arr_attn_mask, arr_label_repr, batch_size):
    arr_input_ids_tensor = tensor(arr_input_ids)
    arr_attn_mask_tensor = tensor(arr_attn_mask)
    arr_label_repr_tensor = tensor(arr_label_repr)

    dataset = TensorDataset(arr_input_ids_tensor, arr_attn_mask_tensor, arr_label_repr_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def preprocessing(csv_file, tokenizer, batch_size):
    """
    @return dataloader (torch.utils.data.DataLoader): dataloader
    """
    sequences, labels = get_sequential_labelling(csv_file)
    arr_input_ids = []
    arr_attention_mask = []
    arr_labels = []
    for seq, label in zip(sequences, labels):
        input_ids, attention_mask, label_repr = process_sequence_and_label(seq, label, tokenizer)
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_labels.append(label_repr)

    tensor_dataloader = create_dataloader(arr_input_ids, arr_attention_mask, arr_labels, batch_size)
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
        pred_prom = output['prom']
        pred_ss = output['ss']
        pred_polya = output['polya']
    return model

def train(model, optimizer, scheduler, train_dataloader, epoch_size, batch_size, log_path, save_model_path, device='cpu', remove_old_model=False, training_counter=0):
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
    log_file.write('epoch,step,loss\n')
    model.to(device)
    model.train()
    for i in range(epoch_size):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids, attention_mask, label = tuple(t.to(device) for t in batch)
            model.zero_grad()
            pred = model(input_ids, attention_mask)
            loss_batch = None
            for p, t in zip(pred, label):
                loss = model.loss_function(p, t)
                if loss_batch == None:
                    loss_batch = loss
                else:
                    loss_batch += loss
            if model.loss_strategy == "average":
                loss_batch = loss_batch / batch_size
            log_file.write("{},{},{}\n".format(i, step, loss_batch))
            loss_batch.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
        #endfor batch

        # After an epoch, save model state.
        save_model_state_dict(model, save_model_path, "epoch-{}.pth".format(i+training_counter))
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