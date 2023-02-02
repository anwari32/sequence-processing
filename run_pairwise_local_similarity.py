import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import pairwise2

training_data = pd.read_csv(os.path.join("error_analysis", "data-comparison", "training_data.csv"))
validation_data = pd.read_csv(os.path.join("error_analysis", "data-comparison", "validation_data.csv"))
test_data = pd.read_csv(os.path.join("error_analysis", "data-comparison", "test_data.csv"))
workdir = os.path.join("error_analysis", "data-comparison")
test_prediction_data = pd.read_csv(os.path.join("prediction", "error_analysis_log_sorted.csv"))

test_similarity_scores = []
for i, r in tqdm(test_prediction_data.iterrows(), total=test_data.shape[0], desc="Processing"):
    e_values = []
    for j, s in training_data.iterrows():
        query = r["original_sequence"]
        subject = s["sequence"]
        alignments = pairwise2.align.localxx(query, subject)
        alignment_score = np.max([a.score for a in alignments])
        e_values.append(alignment_score)

    e_values_str = [str(a) for a in e_values]
    e_values_str = " ".join(e_values_str)
    test_similarity_scores.append(e_values_str)
    
ndf = pd.DataFrame(data={
    "sequence": test_prediction_data["original_sequence"],
    "average_f1": test_prediction_data["average f1"],
    "similarity_score": test_similarity_scores,
})
ndf.to_csv(
    os.path.join(workdir, "test_v_train_comparison.csv")
)