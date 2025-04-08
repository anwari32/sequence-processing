import os
from skbio.alignment import global_pairwise_align_nucleotide
from skbio.sequence import DNA
import pandas as pd
import numpy as np
from tqdm import tqdm
from getopt import getopt
import sys

if __name__ == "__main__":
    opts, args = getopt(sys.argv[1:], "t:b:d:", ["test-data=", "base-data=", "dest-data="])
    output = {}
    for o, a in opts:
        if o in ["-t", "--test-data"]:
            output["test-data"] = a
        elif o in ["-b", "--base-data"]:
            output["base-data"] = a
        elif o in ["-d", "--dest-data"]:
            output["dest-data"] = a
        else:
            raise ValueError(f"option {o} not recognized")

    # prediction_log_dir = os.path.join("prediction")
    # prediction_log_file = os.path.join(prediction_log_dir, "dataframe-F1 Score=1.csv")
    prediction_log_file = output["test-data"]

    # data_dir = os.path.join("error-analysis", "data-comparison")
    # test_data = os.path.join(data_dir, "test_data.csv")
    # validation_data = os.path.join(data_dir, "validation_data.csv")
    # training_data = os.path.join(data_dir, "training_data.csv")
    training_data = output["base-data"]
    output_dir = os.path.join("error-analysis", "alignment", "neddleman-by-sequence")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    prediction_df = pd.read_csv(prediction_log_file)
    training_df = pd.read_csv(training_data)
    test_water_scores = []
    # test_water_scores_str = []
    for i, r in tqdm(prediction_df.iterrows(), total=prediction_df.shape[0], desc="Running Water Algorithm"):
        test_seq = r["original_sequence"]
        water_scores = []
        for j, s in training_df.iterrows():
            training_seq = s["sequence"]
            alignment, score, start_end_positions = global_pairwise_align_nucleotide(
                DNA(test_seq), 
                DNA(training_seq)
            )
            water_scores.append(score)        
        # water_scores_str = " ".join([str(a) for a in water_scores])
        # test_water_scores_str.append(water_scores_str)
        test_water_scores.append(water_scores)
    
    numpy_test_water_scores = np.array(test_water_scores)
    print(f"Comparison object shape {numpy_test_water_scores.shape}")
    
    data = {
        "id": [i for i in range(numpy_test_water_scores.shape[0])],
        "score": [" ".join([str(a) for a in array]) for array in numpy_test_water_scores]
    }
    dest_path = os.path.join(output_dir, output["dest-data"])
    dataframe = pd.DataFrame(data=data)
    dataframe.to_csv(
        dest_path, 
        index=False
    )
    print("Alignment done.")