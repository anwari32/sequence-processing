import os
import pandas as pd
from tqdm import tqdm
from utils.utils import get_sequence_and_label
from skbio.alignment import local_pairwise_align_nucleotide, local_pairwise_align_ssw
from skbio.sequence import DNA
import numpy as np

def parse_score_line(line):
    words = line.split(":")
    score = float(words[1])
    return score

def parse_identity_or_similarity_line(line):
    val = line.split(":")[1].strip()
    vals = val.split(" ")
    vals = vals[0].split("/")
    upper_val = float(vals[0])
    lower_val = float(vals[1])
    score = round(upper_val/lower_val, 3)
    return score

if __name__ == "__main__":
    gene_dir = os.path.join("data", "gene_dir")
    index_dir = os.path.join("index")
    gene_test_index = os.path.join(index_dir, "gene_index.01_test.csv")
    gene_training_index = os.path.join(index_dir, "gene_index.01_train.csv")
    gene_validation_index = os.path.join(index_dir, "gene_index.01_validation.csv")

    water_output_dir = os.path.join("error-analysis", "alignment", "water")
    if not os.path.exists(water_output_dir):
        os.makedirs(water_output_dir, exist_ok=True)

    for p in [gene_test_index, gene_validation_index, gene_training_index]:
        if os.path.exists(p):
            print(f"path found {p}")
        else:
            raise FileNotFoundError(f"path not found at {p}")

    test_df = pd.read_csv(gene_test_index)
    test_sequences = []
    training_df = pd.read_csv(gene_training_index)
    training_sequences = []

    # load gene sequences.
    for i, r in test_df.iterrows():
        path = os.path.join(gene_dir, r["chr"], r["gene"])
        sequences, labels = get_sequence_and_label(path)
        test_sequences.append(
            "".join(sequences)
        )

    for i, r in training_df.iterrows():
        path = os.path.join(gene_dir, r["chr"], r["gene"])
        sequences, labels = get_sequence_and_label(path)
        training_sequences.append(
            "".join(sequences)
        )

    test_water_scores = []

    for test_seq in tqdm(test_sequences, total=len(test_sequences), desc="Comparing"):
        water_scores = []
        for training_seq in training_sequences:
            # r = local_pairwise_align_nucleotide(
            #     DNA(test_seq), 
            #     DNA(training_seq)
            # )
            alignment, score, start_end_positions = local_pairwise_align_ssw(
                DNA(test_seq), 
                DNA(training_seq)
            )
            # water_scores.append(r.score)
            water_scores.append(score)
        test_water_scores.append(water_scores)

    
    numpy_test_water_scores = np.array(test_water_scores)
    print(f"Comparison object shape {numpy_test_water_scores.shape}")
    save_path = os.path.join(water_output_dir, "gene_index_01_comparison.npy")
    if os.path.exists(save_path):
        os.remove(save_path)
    np.save(
        open(os.path.join(water_output_dir, "gene_index_01_comparison.npy"), "wb"),
        numpy_test_water_scores
    )
    
    data = {
        "id": [i for i in range(numpy_test_water_scores.shape[0])],
        "score": [" ".join([str(a) for a in array]) for array in numpy_test_water_scores]
    }
    dataframe = pd.DataFrame(data=data)
    dataframe.to_csv(
        os.path.join(water_output_dir, "smith_waterman_alignment.csv"), index=False
    )
    print("Alignment done.")
