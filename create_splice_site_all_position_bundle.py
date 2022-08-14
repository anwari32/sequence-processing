import pandas as pd
import os
import sys

from tqdm import tqdm
from getopt import getopt
from utils.utils import kmer, str_kmer, is_exists_splice_site_in_sequence

def generate_splice_site_all_pos_bundle(source_gene_dir, bundle_dest_dir, chunk_size, kmer_size):
    """
    `source_gene_dir` - contains directories which each corresponds to chromosome.
    Genes in chromosome folder is in raw format, not kmerized.
    `bundle_dest_dir` - folder where the resulting bundle will be written.
    """
    chr_names = os.listdir(source_gene_dir)
    chr_dirs = [os.path.join(source_gene_dir, a) for a in chr_names]
    chr_dirs = [a for a in chr_dirs if os.path.isdir(a)]

    os.makedirs(bundle_dest_dir, exist_ok=True)
    bundle_path = os.path.join(bundle_dest_dir, "splice_site_all_pos.csv")
    if os.path.exists(bundle_path):
        os.remove(bundle_path)
    
    bundle_file = open(bundle_path, "x")
    bundle_file.write("sequence,label\n")

    for d in tqdm(chr_dirs, total=len(chr_dirs), desc="Processing Chromosome"):
        filenames = os.listdir(d)
        filepaths = [os.path.join(d, a) for a in filenames]
        filepaths = [a for a in filepaths if os.path.isfile(a)]
        
        for f in filepaths:
            df = pd.read_csv(f)
            for i, r in df.iterrows():
                sequence = r["sequence"]
                label = r["label"]
                len_sequence = len(sequence)
                for i in range(0, len_sequence - chunk_size, 1):
                    sublabel = label[i:i+chunk_size]
                    arr_sublabel = kmer(sublabel, kmer_size)
                    if is_exists_splice_site_in_sequence(arr_sublabel):
                        subsequence = sequence[i:i+chunk_size]
                        bundle_file.write(f"{str_kmer(subsequence, kmer_size)},{' '.join(arr_sublabel)}\n")
                    
    bundle_file.close()        

def parse_argv(argv):
    opts, a = getopt(argv, "s:d:c:k:", ["source-gene-dir=", "bundle-destination-dir=", "chunk-size=", "kmer-size="])
    output = {}
    for o, a in opts:
        if o in ["-s", "--source-gene-dir"]:
            output["source-gene-dir"] = a
        elif o in ["-d", "--bundle-destination-dir"]:
            output["bundle-destination-dir"] = a
        elif o in ["-c", "--chunk-size"]:
            output["chunk-size"] = int(a)
        elif o in ["-k", "--kmer-size"]:
            output["kmer-size"] = int(a)
        else:
            raise ValueError(f"Argument {o} not recognized.")
    
    return output

if __name__ == "__main__":
    args = parse_argv(sys.argv[1:])

    generate_splice_site_all_pos_bundle(
        args.get("source-gene-dir"),
        args.get("bundle-destination-dir"),
        args.get("chunk-size"),
        args.get("kmer-size")
    )
