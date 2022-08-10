"""
Merging gene kmerized-contigs.
"""

from genericpath import exists
import os
from this import d
import pandas as pd
from data_preparation import merge_kmer
from tqdm import tqdm

if __name__ == "__main__":
    gene_dir = os.path.join("workspace", "genlab", "seqlab.strand-positive.kmer.stride-510")
    dest_dir = os.path.join("data", "gene_dir_c510_k3")
    chr_dirnames = [d for d in os.listdir(gene_dir) if os.path.isdir(os.path.join(gene_dir, d))]
    for chr_dirname in tqdm(chr_dirnames, total=len(chr_dirnames), desc="Processing"):
        chr_dir = os.path.join(gene_dir, chr_dirname)
        gene_filenames = [f for f in os.listdir(chr_dir) if os.path.isfile(os.path.join(chr_dir, f))]
        for gene_name in gene_filenames:
            gene_file = os.path.join(chr_dir, gene_name)
            df = pd.read_csv(gene_file)
            complete_sequence = []
            complete_label = []
            for i, r in df.iterrows():
                sequence_kmer = r["sequence"].split(" ")
                label_kmer = r["label"].split(" ")
                complete_sequence.append(merge_kmer(sequence_kmer))
                complete_label.append(merge_kmer(label_kmer))
            complete_sequence = "".join(complete_sequence)
            complete_label = "".join(complete_label)
            ndf = pd.DataFrame(data={'sequence': [complete_sequence], 'label': [complete_label]})
            target = os.path.join(dest_dir, chr_dirname, gene_name)
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target), exist_ok=True)
            ndf.to_csv(target, index=False)


