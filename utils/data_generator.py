def _data_generator_mtl(batch_size=1):
    import torch
    import datetime
    import os
    from data_preparation import kmer

    _now = datetime.datetime.now()

    _log_file = os.path.join('logs', 'notebooks', '2022-02.24.csv')
    os.makedirs(_log_file, exist_ok=True)

    #seqs = ["ATGC" * 128, "GATC" * 128, "CCAT" * 128]
    seqs = ["ATGC" * 128, "GACC"  *128, "GTTA" * 128]
    seqs = [' '.join(kmer(s, 3)) for s in seqs]
    prom_labels = [1, 0, 0] #, 0]
    ss_labels = [0, 1, 0]# 0]
    polya_labels = [0, 0, 1] #, 1]

    """
    Initialize BERT tokenizer.
    """
    from transformers import BertTokenizer
    from data_dir import pretrained_3kmer_dir
    import torch
    tokenizer = BertTokenizer.from_pretrained(pretrained_3kmer_dir)

    arr_input_ids = []
    arr_attention_mask = []
    arr_prom_label = []
    arr_ss_label = []
    arr_polya_label = []
    print(f"Data sample {len(seqs)}")
    for i in range(len(seqs)):
        s = seqs[i]
        prom = prom_labels[i]
        ss = ss_labels[i]
        polya = polya_labels[i]

        encoded = tokenizer.encode_plus(text=s, padding="max_length", return_attention_mask=True)
        arr_input_ids.append(encoded.get('input_ids'))
        arr_attention_mask.append(encoded.get('attention_mask'))
        arr_prom_label.append(prom)
        arr_ss_label.append(ss)
        arr_polya_label.append(polya)
    #endfor
    arr_input_ids = torch.tensor(arr_input_ids)
    arr_attention_mask = torch.tensor(arr_attention_mask)
    prom_labels = torch.tensor(arr_prom_label)
    ss_labels = torch.tensor(arr_ss_label)
    polya_labels = torch.tensor(arr_polya_label)

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(arr_input_ids, arr_attention_mask, prom_labels, ss_labels, polya_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def _data_generator_seq2seq():
    from transformers import BertTokenizer
    from data_dir import pretrained_3kmer_dir
    from utils.seq2seq import _create_dataloader, _process_sequence_and_label, _create_dataloader

    """
    Initialize tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_3kmer_dir)

    """
    Create sample data sequential labelling.
    """
    from random import randint
    from data_preparation import kmer
    sequences = ['ATGC' * 128, 'TGAC' * 128, 'GATC' * 128, "AGCC" * 128]
    labels = [['E' if randint(0, 255) % 2 == 0 else '.' for i in range(len(s))] for s in sequences]

    kmer_seq = [' '.join(kmer(sequence, 3)) for sequence in sequences]
    kmer_label = [' '.join(kmer(''.join(label), 3)) for label in labels]

    arr_input_ids = []
    arr_attn_mask = []  
    arr_label_repr = []
    arr_token_type_ids = []
    for seq, label in zip(kmer_seq, kmer_label):
        input_ids, attn_mask, token_type_ids, label_repr = _process_sequence_and_label(seq, label, tokenizer)
        arr_input_ids.append(input_ids)
        arr_attn_mask.append(attn_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_label_repr.append(label_repr)

    dataloader = _create_dataloader(arr_input_ids, arr_attn_mask, arr_token_type_ids, arr_label_repr, batch_size=1)
    return dataloader