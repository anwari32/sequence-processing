import torch
import pandas as pd
from data_preparation import kmer

def get_sequences(csv_path, n_sample=10, random_state=1337):
    r"""
    Get sequence from certain CSV. CSV has header such as 'sequence', 'label_prom', 'label_ss', 'label_polya'.
    """
    df = pd.read_csv(csv_path)
    if (n_sample > 0):
        df = df.sample(n=n_sample, random_state=random_state)
    sequence = list(df['sequence'])
    label_prom = list(df['label_prom'])
    label_ss = list(df['label_ss'])
    label_polya = list(df['label_polya'])

    return sequence, label_prom, label_ss, label_polya

import torch
def preprocessing(data, tokenizer):
    """
    Preprocessing for pretrained BERT.
    @param  data (string): string containing kmers separated by spaces.
    @param  tokenizer (Tokenizer): tokenizer initialized from pretrained values.
    @return input_ids (torch.Tensor): tensor of token ids to be fed to model.
    @return attention_masks (torch.Tensor): tensor of indices (a bunch of 'indexes') specifiying which token needs to be attended by model.
    """
    input_ids = []
    attention_masks = []

    _count = 0
    _len_data = len(data)
    for sequence in data:
        """
        Sequence is 512 characters long.
        """
        _count += 1
        if _count < _len_data:
            print("Seq length = {} [{}/{}]".format(len(sequence.split(' ')), _count, _len_data), end='\r')
        else:
            print("Seq length = {} [{}/{}]".format(len(sequence.split(' ')), _count, _len_data))
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

"""
Initialize tokenizer using BertTokenizer with pretrained weights from DNABert.
"""
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('./pretrained/3-new-12w-0')