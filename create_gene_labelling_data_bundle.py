import os
import pandas as pd
from tqdm import tqdm

def generate_bundle_with_marker(src_index, target_bundle):
    gene_train_index = src_index
    gene_train_bundle_csv = target_bundle
    if os.path.exists(gene_train_bundle_csv):
        os.remove(gene_train_bundle_csv)

    gene_train_bundle = open(gene_train_bundle_csv, "x")
    gene_train_bundle.write("sequence,label,marker\n")

    df = pd.read_csv(gene_train_index)
    marker = True
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc= f"Processing {os.path.basename(src_index)}"):
        chr_dir = r["chr"]
        gene_file = r["gene"]
        gene_name = gene_file.split(".")[0]
        gene_csv = os.path.join(workspace_dir, chr_dir, gene_file)
        gene_df = pd.read_csv(gene_csv)
        for j, k in gene_df.iterrows():
            sequence = k["sequence"]
            label = k["label"]
            gene_train_bundle.write(f"{sequence},{label},{int(marker)}\n")
        
        marker = not marker

    gene_train_bundle.close()

if __name__ == "__main__":
    # Gene sequential labelling.
    # Create 10% and 25% sample of gene indices.
    workspace_dir = os.path.join("workspace", "genlab", "seqlab.strand-positive.kmer.stride-510")
    gene_train_index = os.path.join(workspace_dir, "gene_train_index.csv")
    gene_validation_index = os.path.join(workspace_dir, "gene_validation_index.csv")
    gene_test_index = os.path.join(workspace_dir, "gene_test_index.csv")

    gene_train_index_10 = os.path.join(workspace_dir, "gene_train_index.10.csv")
    gene_validation_index_10 = os.path.join(workspace_dir, "gene_validation_index.10.csv")
    gene_test_index_10 = os.path.join(workspace_dir, "gene_test_index.10.csv")

    gene_train_index_25 = os.path.join(workspace_dir, "gene_train_index.25.csv")
    gene_validation_index_25 = os.path.join(workspace_dir, "gene_validation_index.25.csv")
    gene_test_index_25 = os.path.join(workspace_dir, "gene_test_index.25.csv")

    srcs = [gene_train_index, gene_validation_index, gene_test_index]
    tens = [gene_train_index_10, gene_validation_index_10, gene_test_index_10]
    quarter = [gene_train_index_25, gene_validation_index_25, gene_test_index_25]

    for src, t, q in zip(srcs, tens, quarter):
        src_df = pd.read_csv(src)
        src_10_df = src_df.sample(frac=0.1, random_state=1337)
        src_10_df.to_csv(t, index=False)
        src_25_df = src_df.sample(frac=0.25, random_state=1337)
        src_25_df.to_csv(q, index=False)
    
    workspace_dir = os.path.join("workspace", "genlab", "seqlab.strand-positive.kmer.stride-510")

    srcs = [
        os.path.join(workspace_dir, "gene_train_index.10.csv"),
        os.path.join(workspace_dir, "gene_train_index.25.csv"),
        os.path.join(workspace_dir, "gene_train_index.csv"),
        os.path.join(workspace_dir, "gene_validation_index.10.csv"),
        os.path.join(workspace_dir, "gene_validation_index.25.csv"),
        os.path.join(workspace_dir, "gene_validation_index.csv"),
        os.path.join(workspace_dir, "gene_test_index.10.csv"),
        os.path.join(workspace_dir, "gene_test_index.25.csv"),
        os.path.join(workspace_dir, "gene_test_index.csv"),
    ]
    dests = [
        os.path.join(workspace_dir, "gene_train_index_bundle.10.csv"),
        os.path.join(workspace_dir, "gene_train_index_bundle.20.csv"),
        os.path.join(workspace_dir, "gene_train_index_bundle.csv"),
        os.path.join(workspace_dir, "gene_validation_index_bundle.10.csv"),
        os.path.join(workspace_dir, "gene_validation_index_bundle.20.csv"),
        os.path.join(workspace_dir, "gene_validation_index_bundle.csv"),
        os.path.join(workspace_dir, "gene_test_index_bundle.10.csv"),
        os.path.join(workspace_dir, "gene_test_index_bundle.20.csv"),
        os.path.join(workspace_dir, "gene_test_index_bundle.csv"),
    ]

    # Generate gene bundles based on above.
    for a, b in zip(srcs, dests):
        generate_bundle_with_marker(a, b)

    # Generate gene bundle samples.
    train_sample = os.path.join(workspace_dir, "gene_index.training.sample.csv")
    validation_sample = os.path.join(workspace_dir, "gene_index.validation.sample.csv")
    train_sample_bundle = os.path.join(workspace_dir, "gene_index.training.sample.bundle.csv")
    validation_sample_bundle = os.path.join(workspace_dir, "gene_index.validation.sample.bundle.csv")
    for a, b in zip([train_sample, validation_sample], [train_sample_bundle, validation_sample_bundle]):
        generate_bundle_with_marker(a, b)