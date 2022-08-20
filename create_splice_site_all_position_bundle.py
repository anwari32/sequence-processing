import pandas as pd
import os
import sys

from tqdm import tqdm
from getopt import getopt
from utils.utils import is_exon, is_intron, kmer, str_kmer, is_exists_splice_site_in_sequence

def generate_splice_site_all_pos_bundle(gene_index, source_gene_dir, bundle_dest_dir, chunk_size, kmer_size):
    """
    `source_gene_dir` - contains directories which each corresponds to chromosome.
    Genes in chromosome folder is in raw format, not kmerized.
    `bundle_dest_dir` - folder where the resulting bundle will be written.
    """
    os.makedirs(bundle_dest_dir, exist_ok=True)
    index_name = os.path.basename(gene_index)
    index_name = index_name('.')[0]
    bundle_path = os.path.join(bundle_dest_dir, f"{index_name}_splice_site_all_pos.csv")
    intron_path = os.path.join(bundle_dest_dir, f"{index_name}_intron.csv")
    exon_path = os.path.join(bundle_dest_dir, f"{index_name}_exon.csv")

    if os.path.exists(bundle_path):
        os.remove(bundle_path)
    
    bundle_file = open(bundle_path, "x")
    intron_file = open(intron_path, "x")
    exon_file = open(exon_path, "x")
    bundle_file.write("sequence,label\n")
    intron_file.write("sequence,label\n")
    exon_file.write("sequence,label\n")

    index_df = pd.read_csv(gene_index)
    for _, r in tqdm(index_df.iterrows(), total=index_df.shape[0], desc="Processing Gene"):
        gene_file = os.path.join(source_gene_dir, r["chr"], r["gene"])
        gene_df = pd.read_csv(gene_file)
        for p, q in gene_df.iterrows():
            sequence = q["sequence"]
            label = q["label"]
            len_sequence = len(label)
            for i in range(0, len_sequence - chunk_size + 1, 1):
                sublabel = label[i:i+chunk_size]
                arr_sublabel = kmer(sublabel, kmer_size)
                subsequence = sequence[i:i+chunk_size]
                # if is_exists_splice_site_in_sequence(arr_sublabel):
                if is_intron(arr_sublabel):
                    intron_file.write(f"{str_kmer(subsequence, kmer_size)},{' '.join(arr_sublabel)}\n")
                elif is_exon(arr_sublabel):
                    exon_file.write(f"{str_kmer(subsequence, kmer_size)},{' '.join(arr_sublabel)}\n")
                elif not is_intron(arr_sublabel) and not is_exon(arr_sublabel):
                    bundle_file.write(f"{str_kmer(subsequence, kmer_size)},{' '.join(arr_sublabel)}\n")
                else:
                    raise ValueError(f"Label sequence {arr_sublabel} not recognized.")
                    
    bundle_file.close()        

def parse_argv(argv):
    opts, a = getopt(argv, "i:s:d:c:k:", ["gene-index=", "source-gene-dir=", "bundle-destination-dir=", "chunk-size=", "kmer-size="])
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
        elif o in ["-i", "--gene-index"]:
            output["gene-index"] = a
        else:
            raise ValueError(f"Argument {o} not recognized.")
    
    return output

if __name__ == "__main__":
    args = parse_argv(sys.argv[1:])

    generate_splice_site_all_pos_bundle(
        args.get("gene-index"),
        args.get("source-gene-dir"),
        args.get("bundle-destination-dir"),
        args.get("chunk-size"),
        args.get("kmer-size")
    )
