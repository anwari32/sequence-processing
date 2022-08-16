import os
import pandas as pd

if __name__ == "__main__":

    dirpath = os.path.join("workspace", "seqlab", "seqlab-3")
    gene_bundle_path = os.path.join(dirpath, "gene_bundle.csv")
    gene_train_bundle_path = os.path.join(dirpath, "gene_train_bundle.csv")
    gene_validation_bundle_path = os.path.join(dirpath, "gene_validation_bundle.csv")
    gene_test_bundle_path = os.path.join(dirpath, "gene_test_bundle.csv")

    gene_train_bundle_10_path = os.path.join(dirpath, "gene_train_bundle.10.csv")
    gene_train_bundle_25_path = os.path.join(dirpath, "gene_train_bundle.25.csv")
    gene_train_bundle_sample_path = os.path.join(dirpath, "gene_train_bundle.sample.csv")
    gene_validation_bundle_10_path = os.path.join(dirpath, "gene_validation_bundle.10.csv")
    gene_validation_bundle_25_path = os.path.join(dirpath, "gene_validation_bundle.25.csv")
    gene_validation_bundle_sample_path = os.path.join(dirpath, "gene_validation_bundle.sample.csv")
    gene_test_bundle_10_path =os.path.join(dirpath, "gene_test_bundle.10.csv")
    gene_test_bundle_25_path =os.path.join(dirpath, "gene_test_bundle.25.csv")
    gene_test_bundle_sample_path =os.path.join(dirpath, "gene_test_bundle.sample.csv")

    for p, q, r, s in zip(
        [gene_train_bundle_path, gene_validation_bundle_path, gene_test_bundle_path],
        [gene_train_bundle_10_path, gene_validation_bundle_10_path, gene_test_bundle_10_path],
        [gene_train_bundle_25_path, gene_validation_bundle_25_path, gene_test_bundle_25_path],
        [gene_train_bundle_sample_path, gene_validation_bundle_sample_path, gene_test_bundle_sample_path]
    ):
        source_df = pd.read_csv(p)
        p10_df = source_df.sample(frac=0.1, random_state=1337)
        p25_df = source_df.sample(frac=0.25, random_state=1337)
        psample_df = source_df.sample(frac=0.0001, random_state=1337)
        p10_df.to_csv(q, index=False)
        p25_df.to_csv(r, index=False)
        psample_df.to_csv(s, index=False)
