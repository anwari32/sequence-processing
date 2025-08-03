import os 
import pandas as pd

if __name__ == "__main__":
    dirpath = os.path.join("workspace", "seqlab", "seqlab-3")
    gene_bundle_path = os.path.join(dirpath, "gene_bundle.csv")
    gene_train_bundle_path = os.path.join(dirpath, "gene_train_bundle.csv")
    gene_validation_bundle_path = os.path.join(dirpath, "gene_validation_bundle.csv")
    gene_test_bundle_path = os.path.join(dirpath, "gene_test_bundle.csv")

    df = pd.read_csv(gene_bundle_path)
    train_df = df.sample(frac=0.8, random_state=1337)
    validation_df = df.drop(train_df.index)
    test_df = validation_df.sample(frac=0.5, random_state=1337)
    validation_df = validation_df.drop(test_df.index)
    train_df.to_csv(gene_train_bundle_path, index=False)
    validation_df.to_csv(gene_validation_bundle_path, index=False)
    test_df.to_csv(gene_test_bundle_path, index=False)