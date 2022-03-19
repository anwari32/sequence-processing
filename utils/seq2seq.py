import json
from models.seq2seq import DNABERTSeq2Seq
from torch.optim import AdamW
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from data_preparation import str_kmer

def init_seq2seq_model(config: json):
    model = DNABERTSeq2Seq(config)
    return model

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

Label_Begin = '[CLS]'
Label_End = '[SEP]'
Label_Pad = '[PAD]'
Label_Dictionary = {
    '[CLS]': 0, #_create_one_hot_encoding(0, 10),
    '[SEP]': 1,
    '[PAD]': 2, #_create_one_hot_encoding(9, 10)
    '...': 3, #_create_one_hot_encoding(1, 10),
    '..E': 4, #_create_one_hot_encoding(2, 10),
    '.E.': 5, #_create_one_hot_encoding(3, 10),
    'E..': 6, #_create_one_hot_encoding(4, 10),
    '.EE': 7, #_create_one_hot_encoding(5, 10),
    'EE.': 8, #_create_one_hot_encoding(6, 10),
    'E.E': 9, #_create_one_hot_encoding(7, 10),
    'EEE': 10, #_create_one_hot_encoding(8, 10),
}

from models.seq2seq import DNABERTSeq2Seq

def _get_sequential_labelling(csv_file, do_kmer=False, kmer_size=3):
    df = pd.read_csv(csv_file)
    sequences = list(df['sequence'])
    labels = list(df['label'])
    if do_kmer:
        sequences = [str_kmer(s, kmer_size) for s in sequences]
        labels = [str_kmer(s, kmer_size) for s in labels]
    return sequences, labels

def _process_label(label_sequences, label_dict=Label_Dictionary):
    """
    @param  label_sequences (string): string containing label in kmers separated by spaces. i.e. 'EEE E.. ...'.
    @param  label_dict (map): object to map each kmer into number. Default is `Label_Dictionary`.
    @return (array of integer)
    """
    arr_label_sequence = label_sequences.strip().split(' ')
    label_length = len(arr_label_sequence)
    if label_length < 510:
        delta = 510 - label_length
        for d in range(delta):
            arr_label_sequence.append(Label_Pad)
    label = ['[CLS]']
    label.extend(arr_label_sequence)
    label.extend(['[SEP]'])
    label_kmers = [label_dict[k] for k in label]
    return label_kmers

def _process_sequence_and_label(sequence, label, tokenizer):
    encoded = tokenizer.encode_plus(text=sequence, return_attention_mask=True, return_token_type_ids=True, padding="max_length")
    input_ids = encoded.get('input_ids')
    attention_mask = encoded.get('attention_mask')
    token_type_ids = encoded.get('token_type_ids')
    label_repr = _process_label(label)
    return input_ids, attention_mask, token_type_ids, label_repr

def _create_dataloader(arr_input_ids, arr_attn_mask, arr_token_type_ids, arr_label_repr, batch_size):
    arr_input_ids_tensor = tensor(arr_input_ids)
    arr_attn_mask_tensor = tensor(arr_attn_mask)
    arr_token_type_ids_tensor = tensor(arr_token_type_ids)
    arr_label_repr_tensor = tensor(arr_label_repr)

    dataset = TensorDataset(arr_input_ids_tensor, arr_attn_mask_tensor, arr_token_type_ids_tensor, arr_label_repr_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def preprocessing(csv_file, tokenizer, batch_size, do_kmer=False, kmer_size=3):
    """
    Create dataloader from csv file.
    @param  csv_file (str):
    @param  tokenizer (BertTokenizer):
    @param  batch_size (int | None -> 1):
    @param  kmer_size (int | None -> 3):
    @return dataloader (torch.utils.data.DataLoader): dataloader
    """
    sequences, labels = _get_sequential_labelling(csv_file, do_kmer=do_kmer, kmer_size=kmer_size)
    arr_input_ids = []
    arr_attention_mask = []
    arr_token_type_ids = []
    arr_labels = []
    for seq, label in zip(sequences, labels):
        input_ids, attention_mask, token_type_ids, label_repr = _process_sequence_and_label(seq, label, tokenizer)
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_labels.append(label_repr)

    tensor_dataloader = _create_dataloader(arr_input_ids, arr_attention_mask, arr_token_type_ids, arr_labels, batch_size)
    return tensor_dataloader
