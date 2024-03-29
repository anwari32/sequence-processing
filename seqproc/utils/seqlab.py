from datetime import datetime
import json
from transformers import BertTokenizer
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

from .utils import chunk_string, kmer, str_kmer
from tqdm import tqdm
import os

Label_Begin = '[CLS]'
Label_End = '[SEP]'
Label_Pad = '[PAD]'
Label_Pad = 'III'
Label_Dictionary = {
    '[CLS]': -100,
    '[SEP]': -100,
    '[PAD]': -100, 
    'III': -100,   
    'iii': 0,
    'iiE': 1,
    'iEi': 2,
    'Eii': 3,
    'iEE': 4,
    'EEi': 5,
    'EiE': 6,
    'EEE': 7,
}

label2id = {
    'iii': 0,
    'iiE': 1,
    'iEi': 2,
    'Eii': 3,
    'iEE': 4,
    'EEi': 5,
    'EiE': 6,
    'EEE': 7,
}

id2label = {
    0: "iii",
    1: "iiE",
    2: "iEi",
    3: "Eii",
    4: "iEE",
    5: "EEi",
    6: "EiE",
    7: "EEE",
}

Index_Dictionary = {
    0: "iii",
    1: "iiE",
    2: "iEi",
    3: "Eii",
    4: "iEE",
    5: "EEi",
    6: "EiE",
    7: "EEE",
    -100: "[CLS]/[SEP]/[III]"
}

splice_site_ids = [1, 3, 4, 5]
all_label_ids = [0] + splice_site_ids + [7]
NUM_LABELS = 8

def id2token(id):
    return Index_Dictionary[id]

def token2id(token):
    return Label_Dictionary[token]

def convert_ids_to_tokens(ids):
    return [id2token(id) for id in ids]

def convert_tokens_to_ids(tokens):
    return [token2id(token) for token in tokens]

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

def _process_sequence(sequence, tokenizer):
    """
    @param  sequence (string)
    @param  tokenizer (BertTokenizer)
    return input_ids, attention_mask, token_type_ids
    """
    encoded = tokenizer.encode_plus(text=sequence, return_attention_mask=True, return_token_type_ids=True, padding="max_length", max_length=512)
    input_ids = encoded.get('input_ids')
    attention_mask = encoded.get('attention_mask')
    token_type_ids = encoded.get('token_type_ids')
    return input_ids, attention_mask, token_type_ids

def _process_sequence_and_label(sequence, label, tokenizer):
    input_ids, attention_mask, token_type_ids, label_repr = None, None, None, None
    input_ids, attention_mask, token_type_ids = _process_sequence(sequence, tokenizer)
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

def preprocessing(csv_file: str, tokenizer, batch_size, do_kmer=True, kmer_size=3):
    """
    Create dataloader from csv file. CSV file must contain column ``sequence`` and ``label``.
    Sequence in ``sequence`` must be in regular format, not kmer format.
    @param  csv_file (str): path to CSV file.
    @param  tokenizer (BertTokenizer):
    @param  batch_size (int | None -> 1):
    @param  kmer_size (int | None -> 3):
    @return dataloader (torch.utils.data.DataLoader): dataloader
    """
    start = datetime.now()
    sequences, labels = _get_sequential_labelling(csv_file, do_kmer=do_kmer, kmer_size=kmer_size)
    arr_input_ids = []
    arr_attention_mask = []
    arr_token_type_ids = []
    arr_labels = []
    for seq, label in tqdm(zip(sequences, labels), total=len(sequences), desc="Preprocessing"):
    # for seq, label in zip(sequences, labels):
        input_ids, attention_mask, token_type_ids, label_repr = _process_sequence_and_label(seq, label, tokenizer)
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_labels.append(label_repr)

    tensor_dataloader = _create_dataloader(arr_input_ids, arr_attention_mask, arr_token_type_ids, arr_labels, batch_size)
    end = datetime.now()
    print(f"Preprocessing {csv_file} is finished. Time elapsed {end - start}")
    return tensor_dataloader

def preprocessing_kmer(csv_file: str, tokenizer: BertTokenizer, batch_size, disable_tqdm=False) -> DataLoader:
    """
    Process sequence and label from ``csv_file`` which are already in kmer format. \n
    e.q.    sequence    -> `AAG AGG GGC GCG CGA ...` (kmer format)
            label       -> `iii iiE EEE EEi Eii ...` (kmer format)

    @param  csv_file (str)
    @param  tokenizer (BertTokenizer)
    @param  batch_size (int)
    """
    df = pd.read_csv(csv_file)
    sequences = list(df["sequence"])
    labels = list(df["label"])
    arr_input_ids, arr_attention_mask, arr_token_type_ids, arr_label_repr = [], [], [], []
    enumerator = None
    if disable_tqdm:
        enumerator = zip(sequences, labels)
    else:
        fname = os.path.basename(csv_file).split('.')[:-1]
        fname = ' '.join(fname)
        enumerator = tqdm(zip(sequences, labels), total=df.shape[0], desc=f"Preparing Data {fname}")
    for seq, label in enumerator:
        input_ids, attention_mask, token_type_ids, label_repr = _process_sequence_and_label(seq, label, tokenizer)
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_label_repr.append(label_repr)
    
    tensor_dataloader = _create_dataloader(arr_input_ids, arr_attention_mask, arr_token_type_ids, arr_label_repr, batch_size)
    return tensor_dataloader

def preprocessing_gene_kmer(csv_file: str, tokenizer: BertTokenizer, batch_size, disable_tqdm=False) -> DataLoader:
    # raise NotImplementedError("Not yet implemented.")
    df = pd.read_csv(csv_file)
    sequences = list(df["sequence"])
    labels = list(df["label"])
    markers = list(df["marker"])
    arr_input_ids, arr_attention_mask, arr_token_type_ids, arr_label_repr, arr_marker = [], [], [], [], []
    enumerator = None
    if disable_tqdm:
        enumerator = zip(sequences, labels, markers)
    else:
        fname = os.path.basename(csv_file).split('.')[:-1]
        fname = ' '.join(fname)
        enumerator = tqdm(zip(sequences, labels, markers), total=df.shape[0], desc="Preparing Data")
    for seq, label, marker in enumerator:
        input_ids, attention_mask, token_type_ids, label_repr = _process_sequence_and_label(seq, label, tokenizer)
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_label_repr.append(label_repr)
        arr_marker.append(marker)

    arr_input_ids_tensor = tensor(arr_input_ids)
    arr_attention_mask_tensor = tensor(arr_attention_mask)
    arr_token_type_ids = tensor(arr_token_type_ids)
    arr_label_repr_tensor = tensor(arr_label_repr)
    arr_marker_tensor = tensor(arr_marker)

    dataset = TensorDataset(arr_input_ids_tensor, arr_attention_mask_tensor, arr_token_type_ids, arr_label_repr_tensor, arr_marker_tensor)
    dataloader = DataLoader(dataset, batch_size=1)
    
    return dataloader

def preprocessing_whole_sequence(csv_file: str, tokenizer: BertTokenizer, batch_size=1, dense=False, disable_tqdm=False) -> DataLoader:
    r"""
    Creates dataloader based-on massive dataset.
    * :attr:`csv_file`
    * :attr:`tokenizer`
    * :attr:`batch_size`
    * :attr:`dense` - bool | None -> False
    * :attr:`disable_tqdm`
    """

    # Gene csv should contain only one row, but if it contains more
    # joining sequence and label if gene dataframe contains more than one row.
    complete_sequence = ""
    complete_label = ""
    df = pd.read_csv(csv_file)
    for i, r in df.iterrows():
        complete_sequence = f"{complete_sequence}{r['sequence']}"
        complete_label = f"{complete_label}{r['label']}"

    arr_input_ids = []
    arr_attention_mask = []
    arr_token_type_ids = []
    arr_label_repr = []

    # Break long sequence into shorter sequences with kmer method.
    chunked_sequences = []
    chunked_labels = []
    complete_sequence_kmers = kmer(complete_sequence, 3, 1)
    complete_label_kmers = kmer(complete_label, 3, 1)
    if dense:
        chunked_sequences = kmer(complete_sequence_kmers, 510, 1)
        chunked_labels = kmer(complete_label_kmers, 510, 1)
    else:
        chunked_sequences = chunk_string(complete_sequence_kmers, 510)
        chunked_labels = chunk_string(complete_label_kmers, 510)
    
    chunked_sequences = [' '.join(s) for s in chunked_sequences]
    chunked_labels = [' '.join(s) for s in chunked_labels]
    for seq, lab in zip(chunked_sequences, chunked_labels):
        subsequence_kmer = seq
        sublabel_kmer = lab
        input_ids, attention_mask, token_type_ids, label_repr = None, None, None, None
        try:
            input_ids, attention_mask, token_type_ids, label_repr = _process_sequence_and_label(subsequence_kmer, sublabel_kmer, tokenizer)
        except KeyError:
            print(f"Key Error {subsequence_kmer} \n {sublabel_kmer}")
            print(f"at file {csv_file}")
        
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_label_repr.append(label_repr)

    dataset = TensorDataset(
        tensor(arr_input_ids),
        tensor(arr_attention_mask),
        tensor(arr_token_type_ids),
        tensor(arr_label_repr)
    )
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader



