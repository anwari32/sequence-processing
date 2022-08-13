# Gene labelling: create bundle with marker that indicates a set of sequence belong to one long sequence.
from multiprocessing.sharedctypes import Value
import os
import pandas as pd
from tqdm import tqdm
import sys
from getopt import getopt

def generate_bundle_with_marker(src_index, target_bundle, gene_dir):
    gene_train_index = src_index
    gene_train_bundle_csv = target_bundle
    if os.path.exists(gene_train_bundle_csv):
        os.remove(gene_train_bundle_csv)

    target_bundle_dir = os.path.dirname(target_bundle)
    os.makedirs(target_bundle_dir, exist_ok=True)

    gene_train_bundle = open(gene_train_bundle_csv, "x")
    gene_train_bundle.write("sequence,label,marker\n")

    df = pd.read_csv(gene_train_index)
    marker = True
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc= f"Processing {os.path.basename(src_index)}"):
        chr_dir = r["chr"]
        gene_file = r["gene"]
        gene_csv = os.path.join(gene_dir, chr_dir, gene_file)
        gene_df = pd.read_csv(gene_csv)
        for j, k in gene_df.iterrows():
            sequence = k["sequence"]
            label = k["label"]
            gene_train_bundle.write(f"{sequence},{label},{int(marker)}\n")
        
        marker = not marker

    gene_train_bundle.close()

def parse_args(argv):
    opts, args = getopt(argv, "i:d:g:", ["index-dir=", "destination-dir=", "gene-dir="])
    output = {}
    for o, a in opts:
        if o in ["-i", "--index-dir"]:
            output["index-dir"] = a
        elif o in ["-d", "--destination-dir"]:
            output["destination-dir"] = a
        elif o in ["-g", "--gene-dir"]:
            output["gene-dir"] = a
        else:
            raise ValueError(f"Option {o} not recognized.")
    return output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    # index_dir = os.path.join("index")
    index_dir = args.get("index-dir")
    srcs = [
        os.path.join(index_dir, "gene_train_index.10.csv"),
        os.path.join(index_dir, "gene_train_index.25.csv"),
        os.path.join(index_dir, "gene_train_index.csv"),
        os.path.join(index_dir, "gene_validation_index.10.csv"),
        os.path.join(index_dir, "gene_validation_index.25.csv"),
        os.path.join(index_dir, "gene_validation_index.csv"),
        os.path.join(index_dir, "gene_test_index.10.csv"),
        os.path.join(index_dir, "gene_test_index.25.csv"),
        os.path.join(index_dir, "gene_test_index.csv"),
    ]
    # workspace_dir = os.path.join("workspace", "genlab", "genlab-3")
    workspace_dir = args.get("destination-dir")
    dests = [
        os.path.join(workspace_dir, "gene_train_index_bundle.10.csv"),
        os.path.join(workspace_dir, "gene_train_index_bundle.25.csv"),
        os.path.join(workspace_dir, "gene_train_index_bundle.csv"),
        os.path.join(workspace_dir, "gene_validation_index_bundle.10.csv"),
        os.path.join(workspace_dir, "gene_validation_index_bundle.25.csv"),
        os.path.join(workspace_dir, "gene_validation_index_bundle.csv"),
        os.path.join(workspace_dir, "gene_test_index_bundle.10.csv"),
        os.path.join(workspace_dir, "gene_test_index_bundle.25.csv"),
        os.path.join(workspace_dir, "gene_test_index_bundle.csv"),
    ]
    # Generate gene bundles based on above.
    import os

    # gene_dir = os.path.join("data", "gene_dir_c510_k3")
    gene_dir = args.get("gene-dir")
    for a, b in zip(srcs, dests):
        generate_bundle_with_marker(a, b, gene_dir)