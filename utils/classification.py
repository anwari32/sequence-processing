import pandas as pd
import os
from .utils import kmer, str_kmer
import torch
from tqdm import tqdm
from transformers import BertTokenizer

def extract_acceptor_donor_exon_intron(
    gene_index: str,
    gene_dir: str,
    dest_dir: str,
    flanking_range:int=256,
    ) -> None:
    r"""
    Extracts donor, acceptor, intron, and exon pattern from gene sequence.
    Since intron and exon are obviously large, this procedur will likely to generate
    large files.
    """

    gene_index_df = pd.raed_csv(gene_index)
    gene_files = []
    for _, r in gene_index_df.iterrows():
        chr_dir = r["chr"]
        gene_file = r["gene"]
        gene_path = os.path.join(gene_dir, chr_dir, gene_file)
        gene_files.append(gene_path)
    
    donor_pattern = f"{'E'*flanking_range}{'i'*flanking_range}"
    acceptor_pattern = f"{'i'*flanking_range}{'E'*flanking_range}"
    intron_pattern = f"{'i'*2*flanking_range}"
    exon_pattern = f"{'E'*2*flanking_range}"

    # sequence contains sequence of bases.
    # label contains sequential label of sequence.
    # flag contains label indicating donor, acceptor, intron, or exon (0, 1, 2, 3)
    sequences = []
    labels = []
    classes = []

    for fpath in gene_files:
        df = pd.read_csv(fpath)
        for _, r in df.iterrows():
            sequence_kmers = kmer(r["sequence"], 512, 1)    
            label_kmers = kmer(r["label"], 512, 1)

            for i, j in zip(sequence_kmers, label_kmers):
                sequences.append(i)
                labels.append((j))

                if i == donor_pattern:
                    classes.append(0)
                elif i == acceptor_pattern:
                    classes.append(1)
                elif i == intron_pattern:
                    classes.append(2)
                elif i == exon_pattern:
                    classes.append(3)
                else:
                    classes.append(-1) # Sequence pattern not recognized.
                    
    dest_df = pd.DataFrame({
        'sequence': sequences,
        'label': labels,
        'class': classes
    })
    dest_df.to_csv(os.path.join(dest_dir, "dataset.csv"))
    

def create_dataloader(
    csv_file: str, 
    tokenizer: BertTokenizer,
    batch_size: int=1,
):
    r"""
    Create dataloader.
    """
    if not tokenizer:
        raise ValueError("Tokenizer not initialized.")
    
    df = pd.read_csv(csv_file)
    arr_input_ids = []
    arr_attention_mask = []
    arr_token_type_ids = []
    arr_class = []
    for _, r in df.iterrows():
        sequence = r["sequence"]
        label = r["class"] # Not 'label' because 'label' contains sequential label.

        sequence_kmer = str_kmer(sequence)
        encoded = tokenizer.encode_plus(sequence_kmer, padding="max_length", return_attention_mask=True)
        input_ids = encoded.get('input_ids')
        attention_mask = encoded.get('attention_mask')
        token_type_ids = encoded.get('token_type_ids')
        arr_input_ids.append(input_ids)
        arr_attention_mask.append(attention_mask)
        arr_token_type_ids.append(token_type_ids)
        arr_class.append(label)

    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(arr_input_ids),
        torch.Tensor(arr_attention_mask),
        torch.Tensor(arr_token_type_ids),
        torch.Tensor(arr_class)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

