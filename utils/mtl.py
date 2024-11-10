from datetime import datetime
from pathlib import Path, PureWindowsPath
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
import os
from . import str_kmer

tokenizer = BertTokenizer.from_pretrained(pretrained_3kmer_dir)

def create_dataloader_from_csv(src_file, tokenizer, batch_size=1):
    """
    @param  src_file (string): path to CSV file. Sequence in file already in kmer form.
    @param  tokenizer (BertTokenizer): instance of BertTokenizer.
    @param  batch_size (int = 1): batch size.
    """

    df = pd.read_csv(src_file)

    arr_input_ids = []
    arr_attn_mask = []
    arr_token_type_ids = []
    arr_label_prom = []
    arr_label_ss = []
    arr_label_polya = []
    for i, r in tqdm(df.iterrows(), total=df.shape[0]):
        sent = r["sequence"]
        encoded = tokenizer.encode_plus(
            text=sent, 
            padding="max_length", 
            max_length=512, 
            add_special_tokens=True, 
            return_token_type_ids=True, 
            return_attention_mask=True
        )
        input_ids = encoded.get("input_ids")
        attn_mask = encoded.get("attention_mask")
        token_type_ids = encoded.get("token_type_ids")

        arr_input_ids.append(input_ids)
        arr_attn_mask.append(attn_mask)
        arr_token_type_ids.append(token_type_ids)

        arr_label_prom.append(torch.tensor([int(r["label_prom"])]))
        arr_label_ss.append(torch.tensor([int(r["label_ss"])]))
        arr_label_polya.append(torch.tensor([int(r["label_polya"])]))

    arr_input_ids = torch.tensor(arr_input_ids)
    arr_attn_mask = torch.tensor(arr_attn_mask)
    arr_token_type_ids = torch.tensor(arr_token_type_ids)
    arr_label_prom = torch.tensor(arr_label_prom)
    arr_label_ss = torch.tensor(arr_label_ss)
    arr_label_polya = torch.tensor(arr_label_polya)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, arr_token_type_ids, arr_label_prom, arr_label_polya, arr_label_polya)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def get_sequences(csv_path: str, n_sample=10, random_state=1337, do_kmer=False):
    r"""
    Get sequence from certain CSV. CSV has header such as 'sequence', 'label_prom', 'label_ss', 'label_polya'.
    @param      csv_path (string): path to csv file.
    @param      n_sample (int): how many instance are retrieved from CSV located in `csv_path`.
    @param      random_state (int): random seed for randomly retriving `n_sample` instances.
    @param      do_kmer (bool): determine whether do kmer for sequences.
    @return     (list, list, list, list): sequence, label_prom, label_ss, label_polya.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError("File {} not found.".format(csv_path))
    df = pd.read_csv(csv_path)
    if (n_sample > 0):
        df = df.sample(n=n_sample, random_state=random_state)
    sequence = list(df['sequence'])
    sequence = [str_kmer(s, 3) for s in sequence] if do_kmer == True else sequence
    label_prom = list(df['label_prom'])
    label_ss = list(df['label_ss'])
    label_polya = list(df['label_polya'])

    return sequence, label_prom, label_ss, label_polya

def prepare_data(data, tokenizer: BertTokenizer):
    """
    Preprocessing for pretrained BERT.
    @param      data (array of string): array of string, each string contains kmers separated by spaces.
    @param      tokenizer (Tokenizer): tokenizer initialized from pretrained values.
    @return     input_ids, attention_masks (tuple of torch.Tensor): tensor of token ids to be fed to model,
                tensor of indices (a bunch of 'indexes') specifiying which token needs to be attended by model.
    """
    input_ids = []
    attention_masks = []
    _count = 0
    _len_data = len(data)
    for sequence in tqdm(data, total=_len_data, desc="Preparing Data"):
        """
        Sequence is 512 characters long.
        """
        _count += 1
        #if _count < _len_data:
        #    print("Seq length = {} [{}/{}]".format(len(sequence.split(' ')), _count, _len_data), end='\r')
        #else:
        #    print("Seq length = {} [{}/{}]".format(len(sequence.split(' ')), _count, _len_data))
        encoded_sent = tokenizer.encode_plus(
            text=sequence,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert input_ids and attention_masks to tensor.
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

def preprocessing(csv_file: str, pretrained_tokenizer_path: str, batch_size=2000, n_sample=0, random_state=1337, do_kmer=False):
    """
    @return dataloader (torch.utils.data.DataLoader)
    """
    csv_file = PureWindowsPath(csv_file)
    csv_file = str(Path(csv_file))
    if not os.path.exists(csv_file):
        raise FileNotFoundError("File {} not found.".format(csv_file))
    _start_time = datetime.now()

    bert_path = PureWindowsPath(pretrained_tokenizer_path)
    bert_path = str(Path(bert_path))
    # tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sequences, prom_labels, ss_labels, polya_labels = get_sequences(csv_file, n_sample=n_sample, random_state=random_state, do_kmer=do_kmer)
    arr_input_ids, arr_attn_mask = prepare_data(sequences, tokenizer)
    prom_labels_tensor = torch.tensor(prom_labels)
    ss_labels_tensor = torch.tensor(ss_labels)
    polya_labels_tensor = torch.tensor(polya_labels)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, prom_labels_tensor, ss_labels_tensor, polya_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print("Preparing Dataloader duration {}".format(_elapsed_time))
    return dataloader

def preprocessing_batches(csv_file: str, pretrained_tokenizer_path: str, batch_sizes=[], n_sample=0, random_state=1337, do_kmer=False):
    if len(batch_sizes) == 0:
        raise ValueError(f"Batch sizes cannot be {batch_sizes}")
    csv_file = PureWindowsPath(csv_file)
    csv_file = str(Path(csv_file))
    if not os.path.exists(csv_file):
        raise FileNotFoundError("File {} not found.".format(csv_file))
    _start_time = datetime.now()

    bert_path = PureWindowsPath(pretrained_tokenizer_path)
    bert_path = str(Path(bert_path))
    # tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sequences, prom_labels, ss_labels, polya_labels = get_sequences(csv_file, n_sample=n_sample, random_state=random_state, do_kmer=do_kmer)
    arr_input_ids, arr_attn_mask = prepare_data(sequences, tokenizer)
    prom_labels_tensor = torch.tensor(prom_labels)
    ss_labels_tensor = torch.tensor(ss_labels)
    polya_labels_tensor = torch.tensor(polya_labels)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, prom_labels_tensor, ss_labels_tensor, polya_labels_tensor)
    dataloaders = []
    for batch_size in batch_sizes:
        dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print(f"Preparing Dataloaders duration {_elapsed_time}")
    return dataloaders
    
