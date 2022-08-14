# Read original gene_dir and convert each gene into kmerized gene sequence.

from genericpath import isdir
import os
from utils.utils import chunk_string, str_kmer
import pandas as pd
from tqdm import tqdm
from getopt import getopt
import sys

def create_gene_dir(original_gene_dir, target_gene_dir, chunk_size, kmer_size):
    source_chr_dirs = [a for a in os.listdir(original_gene_dir) if os.path.isdir(os.path.join(original_gene_dir, a))]
    target_chr_dirs = [os.path.join(target_gene_dir, a) for a in source_chr_dirs]
    source_chr_dirs = [os.path.join(original_gene_dir, a) for a in source_chr_dirs]

    for s, d in tqdm(zip(source_chr_dirs, target_chr_dirs), total=len(source_chr_dirs), desc="Creating Gene Dir"):
        source_files = [a for a in os.listdir(s) if os.path.isfile(os.path.join(s, a))]
        dest_files = [os.path.join(d, a) for a in source_files]
        source_files = [os.path.join(s, a) for a in source_files]

        for p, q in zip(source_files, dest_files):
            if os.path.exists(q):
                os.remove(q)
            qdir = os.path.dirname(q)
            if not os.path.exists(qdir):
                os.makedirs(qdir, exist_ok=True)
        
            dest_file = open(q, "x")
            dest_file.write("sequence,label\n")
            df = pd.read_csv(p)
            for i, r in df.iterrows():
                complete_sequence = r["sequence"]
                complete_label = r["label"]

                sequence_chunks = chunk_string(complete_sequence, chunk_size)
                label_chunks = chunk_string(complete_label, chunk_size)

                for seq_chunk, label_chunk in zip(sequence_chunks, label_chunks):
                    seq_chunk_kmer = str_kmer(seq_chunk, kmer_size)
                    label_chunk_kmer = str_kmer(label_chunk, kmer_size)
                    dest_file.write(f"{seq_chunk_kmer},{label_chunk_kmer}\n")
            
            dest_file.close()

def parse(argv):
    opts, args = getopt(argv, "s:d:c:k:", ["source-dir=", "destination-dir=", "chunk-size=", "kmer-size="])
    output = {}
    for o, a in opts:
        if o in ["-s", "--source-dir"]:
            output["source-dir"] = a
        elif o in ["-d", "--destination-dir"]:
            output["destination-dir"] = a
        elif o in ["-c", "--chunk-size"]:
            output["chunk-size"] = int(a)
        elif o in ["-k", "--kmer-size"]:
            output["kmer-size"] = int(a)
        else:
            raise ValueError(f"Argument {o} not recognzed.")
    return output

if __name__ == "__main__":
    args = parse(sys.argv[1:])

    create_gene_dir(
        args.get("source-dir"),
        args.get("destination-dir"),
        args.get("chunk-size"),
        args.get("kmer-size")
    )

            


    
