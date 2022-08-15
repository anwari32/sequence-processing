# Create gene bundle based on given gene index.
from getopt import getopt
import os
import sys
import pandas as pd
from tqdm import tqdm

def parse_args(argv):
    opts, _ = getopt(argv, "i:g:d:", ["index=", "gene-dir=", "dest-dir="])
    output = {}
    for o, a in opts:
        if o in ["-i", "--index"]:
            output["index"] = a
        elif o in ["-g", "--gene-dir"]:
            output["gene-dir"] = a
        elif o in ["-d", "--dest-dir"]:
            output["dest-dir"] = a
        else:
            raise ValueError(f"Argument {o} not recognized.")
    
    return output

def generate_gene_bundle_from_index(index, gene_dir, dest_dir):
    whole_index = index
    bundle_dir = os.path.join(dest_dir)
    whole_bundle = os.path.join(bundle_dir, "gene_bundle.csv")
    for s, d in zip([whole_index], [whole_bundle]):
        df = pd.read_csv(s)
        ddirname = os.path.dirname(d)
        if not os.path.exists(ddirname):
            os.makedirs(ddirname)
        if os.path.exists(d):
            os.remove(d)
        dest_file = open(d, "x")
        dest_file.write("sequence,label\n")
        for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Bundling"):
            chr = r["chr"]
            gene = r["gene"]
            gene_path = os.path.join(gene_dir, chr, gene)
            gene_df = pd.read_csv(gene_path)
            for j, k in gene_df.iterrows():
                dest_file.write(f"{k['sequence']},{k['label']}\n")

        dest_file.close()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    generate_gene_bundle_from_index(
        args.get("index"),
        args.get("gene-dir"),
        args.get("dest-dir")
    )
