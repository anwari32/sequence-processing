from concurrent.futures import process
from msilib import sequence
from torch import nn, tensor
from torch.nn import NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, BertForMaskedLM
import os
import pandas as pd
from tqdm import tqdm

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
    def __init__(self, bert_layer, seq2seq_head, device='cpu'):
        """
        This model uses BERT as its feature extraction layer.
        This BERT layer is initiated from pretrained model which is located at `bert_pretrained_path`.
        @param  bert_pretrained_path (string):
        @param  device (string): default is 'cpu' but you can put 'cuda' if your machine supports cuda.
        @return (DNASeqLabelling): object of this class.
        """
        super().__init__()
        self.bert = bert_layer
        self.seq2seq_head = seq2seq_head
        self.loss_function = nn.NLLLoss()

    def forward(self, input_ids, attention_masks):
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        output = output[0][:,0,:]
        output = self.seq2seq_head(output)
        return output

def initialize_seq2seq(bert_pretrained_path, in_out_dims):
    seq2seq_head = Seq2SeqHead(in_out_dims)
    bert_layer = BertForMaskedLM.from_pretrained(bert_pretrained_path).bert
    model = DNABERTSeq2Seq(bert_layer, seq2seq_head)
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

def init_adamw_optimizer(model_parameters, learning_rate=1e-5, epsilon=1e-6):
    """
    Initialize AdamW optimizer.
    @param  model_parameters:
    @param  learning_rate: Default is 1e-5 so it's small. 
            Change to 2e-4 for fine-tuning purpose (Ji et. al., 2021) or 4e-4 for pretraining (Ji et. al., 2021).
    @param  epsilon: adam epsilon, default is 1e-6 as in DNABERT pretraining (Ji et. al., 2021).
    @return (AdamW object)
    """
    optimizer = AdamW(model_parameters, lr=learning_rate, eps=epsilon)
    return optimizer

def train_and_validation(model, optimizer, scheduler, train_dataloader, validation_dataloader, epoch_size, log_file, save_path, device='cpu'):
    """
    @param  model: BERT derivatives.
    @param  optimizer: optimizer
    @param  scheduler:
    @param  train_dataloader:
    @param  validation_dataloader:
    @param  epoch_size:
    @param  log_file:
    @param  device: default value is 'cpu', can be changed into 'cuda', 'cuda:0' for first cuda-compatible device, 'cuda:1' for second device, etc.
    @return model after training.
    """
    model.to(device)
    model.train()
    for i in range(epoch_size):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids, attention_mask, label = tuple(t.to(device) for t in batch)
            pred = model(input_ids, attention_mask)
            loss = model.loss_function(pred, label)
            loss.backward()
        #endfor batch
    #endfor range
    return model
    