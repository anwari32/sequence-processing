import os
import pandas as pd
from utils.utils import kmer, str_kmer
from tqdm import tqdm

# configurations.
# change these constants to generate desired dataset.
LENGTH = 512
STRIDE = 50

KMER_SIZE = 3
KMER_STRIDE = 1

if __name__ == "__main__":
    gene_dir = os.path.join("data", "gene_dir")
    index_dir = os.path.join("index")
    gene_train_index = os.path.join(index_dir, "gene_index_train.csv")
    gene_test_index = os.path.join(index_dir, "gene_index_test.csv")
    gene_train_index_df = pd.read_csv(gene_train_index)
    gene_test_index_df = pd.read_csv(gene_test_index)

    work_dir = os.path.join("workspace", f"seqlab-stride_{STRIDE}")
    os.makedirs(work_dir, exist_ok=True)
    train_file_path = os.path.join(work_dir, "train_data.csv")
    test_file_path = os.path.join(work_dir, "test_data.csv")
    
    df_list = [gene_train_index_df, gene_test_index_df]
    output_list = [train_file_path, test_file_path]

    for p in [gene_dir, index_dir, gene_train_index, gene_test_index, work_dir]:
        if os.path.exists(p):
            print(f"Path found at {p}")
        else:
            raise FileNotFoundError(f"{p}")

    for df, output_path in zip(df_list, output_list):
        print(f"Generating {output_path}")
        sequence_list = []
        label_list = []
        for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Generating"):
            gene_df = pd.read_csv(os.path.join(gene_dir, r["chr"], r["gene"]))
            for j, s in gene_df.iterrows():
                sequence = s["sequence"]
                sequence_chunks = kmer(sequence, LENGTH, STRIDE)
                sequence_chunks = [str_kmer(chunk, KMER_SIZE, KMER_STRIDE) for chunk in sequence_chunks]
                sequence_list += sequence_chunks
                label = s["label"]
                label_chunks = kmer(label, LENGTH, STRIDE)
                label_chunks = [str_kmer(chunk, KMER_SIZE, KMER_STRIDE) for chunk in label_chunks]
                label_list += label_chunks

        dataframe = pd.DataFrame(data={
            "sequence": sequence_list,
            "label": label_list
        })
        print(f"Data shape {dataframe.shape}")
        dataframe.to_csv(output_path, index=False)