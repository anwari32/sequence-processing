from models.ensemble import Ensemble
from models.dnabert import num_classes
import os
import numpy as np
from utils.metrics import Metrics
import pandas as pd

if __name__ == "__main__":
    pred_dir = os.path.join("prediction")
    baseline_dir = os.path.join(pred_dir, "baseline")
    dnabert_dir = os.path.join(pred_dir, "dnabert")
    bilstm_df = pd.read_csv(
        os.path.join(baseline_dir, "bilstm", "bilstm_log.csv")
    )
    bilstm_pred = []
    for i, r in bilstm_df.iterrows():
        bilstm_pred.append(
            r["sequence"].split(" ")
        )

    bigru_df = pd.read_csv(
        os.path.join(baseline_dir, "bigru", "bigru_log.csv")
    )
    bigru_pred = []
    for i, r in bigru_df.iterrows():
        bigru_pred.append(
            r["sequence"].split(" ")
        )

    dnabert_df = pd.read_csv(
        os.path.join(dnabert_dir, "dnabert_log.csv")
    )
    dnabert_pred = []
    for i, r in dnabert_df.iterrows():
        dnabert_pred.append(
            r["sequence"].split(" ")
        )
    
    ensemble = Ensemble(
        pred_bilstm = np.array(bilstm_pred).flatten(),
        pred_bigru = np.array(bigru_pred).flatten(),
        pred_dnabert = np.array(dnabert_pred).flatten(),
        num_classes=num_classes
    )
    final_prediction = ensemble.consolidate()
