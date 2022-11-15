import os
from skbio.alignment import local_pairwise_align_ssw
from skbio.sequence import DNA
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    data_dir = os.path.join("error-analysis", "data-comparison")
    test_data = os.path.join(data_dir, "test_data.csv")
    validation_data = os.path.join(data_dir, "validation_data.csv")
    training_data = os.path.join(data_dir, "training_data.csv")
    water_output_dir = os.path.join("error-analysis", "alignment", "water-by-sequence")
    if not os.path.exists(water_output_dir):
        os.makedirs(water_output_dir, exist_ok=True)

    for p in [test_data, validation_data, training_data]:
        if os.path.exists(p):
            print(f"path found {p}")
        else:
            raise FileNotFoundError(f"path not found at {p}")


    test_df = pd.read_csv(test_data)
    training_df = pd.read_csv(training_data)
    test_water_scores = []
    for i, r in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Running Water Algorithm"):
        test_seq = r["sequence"]
        water_scores = []
        for j, s in training_df.iterrows():
            training_seq = s["sequence"]
            alignment, score, start_end_positions = local_pairwise_align_ssw(
                DNA(test_seq), 
                DNA(training_seq)
            )        
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