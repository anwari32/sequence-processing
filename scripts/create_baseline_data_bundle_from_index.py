from getopt import getopt
import pandas as pd
from tqdm import tqdm
import os
import sys
import utils

def parse_argv(argv):
    opts, a = getopt(argv, "i:s:d:c:k:", ["gene-index=", "source-gene-dir=", "bundle-destination-dir=", "window-size=", "kmer-size=", "ss-only", "kmer"])
    output = {}
    for o, a in opts:
        if o in ["-s", "--source-gene-dir"]:
            output["source-gene-dir"] = a
        elif o in ["-d", "--bundle-destination-dir"]:
            output["bundle-destination-dir"] = a
        elif o in ["-c", "--window-size"]:
            output["window-size"] = int(a)
        elif o in ["-k", "--stride"]:
            output["stride"] = int(a)
        elif o in ["-i", "--gene-index"]:
            output["gene-index"] = a
        elif o in ["--ss-only"]:
            output["ss-only"] = True
        elif o in ["--kmer"]:
            output["kmer"] = True
        else:
            raise ValueError(f"Argument {o} not recognized.")

    return output

def generate_data(gene_index, source_gene_dir, bundle_dest_dir, window_size, stride, ss_only=False, kmer=False):
    os.makedirs(bundle_dest_dir, exist_ok=True)
    index_name = os.path.basename(gene_index)
    index_name = index_name.split('.')[:-1]
    index_name = '.'.join(index_name)
    bundle_path = os.path.join(bundle_dest_dir, f"{index_name}_ss_all_pos.csv")

    if os.path.exists(bundle_path):
        os.remove(bundle_path)
    
    bundle_file = open(bundle_path, "x")
    bundle_file.write("sequence,label\n")

    index_df = pd.read_csv(gene_index)
    for _, r in tqdm(index_df.iterrows(), total=index_df.shape[0], desc="Processing Gene"):
        gene_file = os.path.join(source_gene_dir, r["chr"], r["gene"])
        gene_df = pd.read_csv(gene_file)
        for p, q in gene_df.iterrows():
            sequence = q["sequence"]
            label = q["label"]
            len_sequence = len(label)
            for i in range(0, len_sequence - window_size + stride, stride):
                sublabel = label[i:i+window_size]
                subsequence = sequence[i:i+window_size]

                if kmer:
                    subsequence =  utils.utils.kmer(subsequence, 3, 1)
                    sublabel = utils.utils.kmer(sublabel, 3, 1)

                if ss_only:
                    is_all_intron = all([t == "iii" for t in sublabel])
                    is_all_exon = all([t == "EEE" for t in sublabel])
                    if (not is_all_intron) and (not is_all_exon):
                        subsequence = " ".join(subsequence)
                        sublabel = " ".join(sublabel)
                        bundle_file.write(f"{subsequence},{sublabel}\n")
                else:
                    subsequence = " ".join(subsequence)
                    sublabel = " ".join(sublabel)
                    bundle_file.write(f"{subsequence},{sublabel}\n")
    
    bundle_file.close()
            
if __name__ == "__main__":
    args = parse_argv(sys.argv[1:])

    generate_data(
        args.get("gene-index"),
        args.get("source-gene-dir"),
        args.get("bundle-destination-dir"),
        args.get("window-size"),
        args.get("stride"),
        args.get("ss-only", False), # by default, extract all sequence.
        args.get("kmer", False)
    )
