import pandas as pd
import os
from tqdm import tqdm
from . import kmer, merge_kmer

donor_pattern = ["EEi", "Eii"]
acceptor_pattern = ["iiE", "iEE"]
default_window_size = 13

def extract_motif(csv_path, dest_dir="motif_analysis", window_size=default_window_size):
    dest_file = os.path.basename(csv_path).split(".")[0:-1]
    dest_file = ".".join(dest_file)
    dest_acceptor_file = f"{dest_file}.acceptor.csv"
    dest_donor_file = f"{dest_file}.donor.csv"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    dest_acceptor_file = os.path.join(dest_dir, dest_acceptor_file)
    dest_donor_file = os.path.join(dest_dir, dest_donor_file)

    donor_pattern = ["EEi", "Eii"]
    acceptor_pattern = ["iiE", "iEE"]
    donor_sequences, donor_predictions, donor_targets, donor_sequence_tokens, donor_prediction_tokens, donor_target_tokens = [], [], [], [], [], []
    acceptor_sequences, acceptor_predictions, acceptor_targets, acceptor_sequence_tokens, acceptor_prediction_tokens, acceptor_target_tokens = [], [], [], [], [], []

    df = pd.read_csv(csv_path)
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting"):
        input_tokens = r["input_tokens"].split(" ")
        prediction_tokens = r["prediction_tokens"].split(" ")
        target_tokens = r["target_tokens"].split(" ")

        arr_i = kmer(input_tokens, window_size, 1)
        arr_j = kmer(prediction_tokens, window_size, 1)
        arr_k = kmer(target_tokens, window_size, 1)

        for i, j, k in zip(arr_i, arr_j, arr_k):
            _j = kmer(j, 2, 1)
            _k = kmer(k, 2, 1)
            if donor_pattern in _j or donor_pattern in _k:
                donor_sequences.append(merge_kmer(i))
                donor_sequence_tokens.append(" ".join(i))
                donor_predictions.append(merge_kmer(j))
                donor_prediction_tokens.append(" ".join(j))
                donor_targets.append(merge_kmer(k))
                donor_target_tokens.append(" ".join(k))
            if acceptor_pattern in _j or acceptor_pattern in _k:
                acceptor_sequences.append(merge_kmer(i))
                acceptor_sequence_tokens.append(" ".join(i))
                acceptor_predictions.append(merge_kmer(j))
                acceptor_prediction_tokens.append(" ".join(j))
                acceptor_targets.append(merge_kmer(k))
                acceptor_target_tokens.append(" ".join(k))
            
    donor_df = pd.DataFrame(data={
        "sequence": donor_sequences,
        "prediction": donor_predictions,
        "target": donor_targets,
        "sequence_tokens": donor_sequence_tokens,
        "prediction_tokens": donor_prediction_tokens,
        "target_tokens": donor_target_tokens
    })
    donor_df.to_csv(dest_donor_file, index=False)
    acceptor_df = pd.DataFrame(data={
        "sequence": acceptor_sequences,
        "prediction": acceptor_predictions,
        "target": acceptor_targets,
        "sequence_tokens": acceptor_sequence_tokens,
        "prediction_tokens": acceptor_prediction_tokens,
        "target_tokens": acceptor_target_tokens
    })
    acceptor_df.to_csv(dest_acceptor_file, index=False)
