from datetime import datetime
import json
from transformers import BertTokenizer
from models.seqlab import DNABERT_SL
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from data_preparation import str_kmer
from tqdm import tqdm

def init_seqlab_model(config: json):
    if not config:
        raise ValueError("Not valid json object.")
    model = DNABERT_SL(config)
    return model


Label_Begin = '[CLS]'
Label_End = '[SEP]'
# Label_Pad = '[PAD]'
Label_Pad = 'III'
#Label_Dictionary = {
#    '[CLS]': 0, #_create_one_hot_encoding(0, 10),
#    '[SEP]': 1,
    # '[PAD]': 2, #_create_one_hot_encoding(9, 10)
#    'III': 2,   # Created III instead of iii for PAD special token. 
                # This is for enabling reading contigs if token is predicted as padding. 
                # #_create_one_hot_encoding(9, 10)
#    'iii': 3,   #_create_one_hot_encoding(1, 10),
#    'iiE': 4,   #_create_one_hot_encoding(2, 10),
#    'iEi': 5,   #_create_one_hot_encoding(3, 10),
#    'Eii': 6,   #_create_one_hot_encoding(4, 10),
#    'iEE': 7,   #_create_one_hot_encoding(5, 10),
#    'EEi': 8,   #_create_one_hot_encoding(6, 10),
#    'EiE': 9,   #_create_one_hot_encoding(7, 10),
#    'EEE': 10,  #_create_one_hot_encoding(8, 10),
#}

Label_Dictionary = {
    '[CLS]': -100, #_create_one_hot_encoding(0, 10),
    '[SEP]': -100,
    # '[PAD]': 2, #_create_one_hot_encoding(9, 10)
    'III': -100,   # Created III instead of iii for PAD special token. 
                # This is for enabling reading contigs if token is predicted as padding. 
                # #_create_one_hot_encoding(9, 10)
    'iii': 0,   #_create_one_hot_encoding(1, 10),
    'iiE': 1,   #_create_one_hot_encoding(2, 10),
    'iEi': 2,   #_create_one_hot_encoding(3, 10),
    'Eii': 3,   #_create_one_hot_encoding(4, 10),
    'iEE': 4,   #_create_one_hot_encoding(5, 10),
    'EEi': 5,   #_create_one_hot_encoding(6, 10),
    'EiE': 6,   #_create_one_hot_encoding(7, 10),
    'EEE': 7,  #_create_one_hot_encoding(8, 10),
}

Index_Dictionary = {
    #0: "[CLS]",
    #1: "[SEP]",
    # 2: "[PAD]",
    #2: "III",       # Use `III` as padding symbol.
    0: "iii",
    1: "iiE",
    2: "iEi",
    3: "Eii",
    4: "iEE",
    5: "EEi",
    6: "EiE",
    7: "EEE"
}

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
    encoded = tokenizer.encode_plus(text=sequence, return_attention_mask=True, return_token_type_ids=True, padding="max_length")
    input_ids = encoded.get('input_ids')
    attention_mask = encoded.get('attention_mask')
    token_type_ids = encoded.get('token_type_ids')
    return input_ids, attention_mask, token_type_ids

def _process_sequence_and_label(sequence, label, tokenizer):
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

def preprocessing_kmer(csv_file: str, tokenizer: BertTokenizer, batch_size) -> DataLoader:
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
    for seq, label in tqdm(zip(sequences, labels), total=df.shape[0], desc="Preparing data "):
        input_ids, attention_mask, token_type_ids, label_repr = _process_sequence_and_label(seq, label, tokenizer)
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_label_repr.append(label_repr)
    
    tensor_dataloader = _create_dataloader(arr_input_ids, arr_attention_mask, arr_token_type_ids, arr_label_repr, batch_size)
    return tensor_dataloader
