# generate balanced dataset from small index.
# need to work in complete index.
# patterns: EEi Eii => EEEiii
#           iiE iEE => iiiEEE

import os
import sys
import pandas as pd
from getopt import getopt
from tqdm import tqdm
from utils.utils import kmer

kmer_donor_pattern = ["EEi", "Eii"]
kmer_acceptor_pattern = ["iiE", "iEE"]
donor_pattern = ["E" for i in range(256)] + ["i" for i in range(256)]
acceptor_pattern = ["i" for i in range(256)] + ["E" for i in range(256)]

def __parse__(argv):
    opts, arguments = getopt(argv, "i:g:l:r:d:c:", ["index=", "gene-dir=", "left-flank=", "right-flank=", "dest=", "chunk-size="])
    args = {}
    for o, a in opts:
        if o in ["-i", "--index"]:
            args["index"] = a
        elif o in ["-g", "--gene-dir"]:
            args["gene-dir"] = a
        elif o in ["-l", "left-flank"]:
            args["left-flank"] = int(a)
        elif o in ["-r", "right-flank"]:
            args["right-flank"] = int(a)
        elif o in ["-d", "--dest"]:
            args["dest"] = str(a)
        elif o in ["-c", "--chunk-size"]:
            args["chunk-size"] = int(a)
        # elif o in ["kmer"]:
        #     args["kmer"] = True
        else:
            raise ValueError(f"keyword {o} not recognized")
    return args

def generate_ss_with_flank(sequence, label, nleft_flank, nright_flank):
    sequences = []
    labels = []
    seq = kmer(sequence, 3, 1)
    lab = kmer(label, 3, 1)
    seq_length = len(seq)
    lab_length = len(lab)
    for i in tqdm(range(0, seq_length-1, 1), desc=f"{sequence[0:10]} ..."):
        sublab = lab[i:i+2]
        if sublab == donor_pattern or sublab == acceptor_pattern:
            start_index = i - nleft_flank
            if start_index < 0:
                start_index = 0
            end_index = i + 2 + nright_flank
            if end_index > seq_length:
                end_index = seq_length
            subseq = seq[start_index:end_index]
            sublab = lab[start_index:end_index]
            sequences.append(" ".join(subseq))
            labels.append(" ".join(sublab))
    print(sequences)
    print(labels)
    return sequences, labels

def generate_ss_all_position(sequence, label, chunk_size):
    sequences, labels = [], []
    seq = kmer(sequence, 3, 1)
    lab = kmer(label, 3, 1)
    seq_length = len(seq)
    lab_length = len(lab)
    if len(seq) <= chunk_size:
        sequences.append(" ".join(seq))
        labels.append(" ".join(lab))
        return sequences, labels

    nflanking = chunk_size - 2
    for i in range(0, seq_length-1, 1):
        sublab = lab[i:i+2]
        if sublab == donor_pattern or sublab == acceptor_pattern:
            for left_flank in [r for r in range(0, nflanking + 1, 1)]:
                right_flank = nflanking - left_flank
                start_index = i - left_flank
                if start_index < 0:
                    start_index = 0
                end_index = i + 2 + right_flank
                if end_index > seq_length:
                    end_index = seq_length
                subseq = seq[start_index: end_index]
                sublab = lab[start_index: end_index]
                sequences.append(" ".join(subseq))
                labels.append(" ".join(sublab))
    
    return sequences, labels

if __name__ == "__main__":

    command_args = sys.argv[1:]
    args = __parse__(command_args)
    
    gene_index_path = args.get("index", False)
    gene_dir = args.get("gene-dir", False)
    nleft_flank = args.get("left-flank", False)
    nright_flank = args.get("right-flank", False)
    chunk_size = args.get("chunk-size", 510)
    dest = args.get("dest", False)

    # gene_index_path = os.path.join("index", "gene_index_train.csv")
    # gene_dir = os.path.join("data", "gene_dir")

    gene_index = pd.read_csv(gene_index_path)
    gene_index = gene_index.sample(frac=0.01)
    sequences = []
    labels = []
    for i, r in gene_index.iterrows():
        chr = r["chr"]
        gene = r["gene"]
        gene_df = pd.read_csv(os.path.join(gene_dir, chr, gene))
        for j, s in tqdm(gene_df.iterrows(), desc=f"[{i + 1}/{gene_index.shape[0]}] {gene}"):
            seq = s["sequence"]
            lab = s["label"]
            subseq = []
            sublab = []
            if nleft_flank and nright_flank:
                # generate ss with certain flanks.
                subseq, sublab = generate_ss_with_flank(seq, lab, nleft_flank, nright_flank)
            else:
                # generate for all possible flanks.
                subseq, sublab = generate_ss_all_position(seq, lab, chunk_size=chunk_size)

            sequences += subseq
            labels += sublab

    dataframe = pd.DataFrame(data={
        "sequence": sequences,
        "label": labels
    })

    dirpath = os.path.dirname(dest)
    filename = os.path.basename(dest)
    os.makedirs(dirpath, exist_ok=True)
    print(f"Writing {dataframe.shape[0]} sequences ...", end="\r")
    dataframe.to_csv(
        os.path.join(dirpath, filename),
        index=False
    )
    print(f"Writing {dataframe.shape[0]} sequences DONE")

    destpath = args.get("dest", os.path.join("workspace", "seqclass", "file.csv"))
    dirpath = os.path.dirname(destpath)
    os.makedirs(dirpath, exist_ok=True)
    dataframe.to_csv(
        os.path.join(destpath),
        index=False
    )

                