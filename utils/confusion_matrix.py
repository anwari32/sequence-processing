from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def filter_df_by_epoch(csv_file, epochs):
    results = []
    df = pd.read_csv(csv_file)
    for e in epochs:
        results.append(df[df["epoch"] == e])
    return results

def create_cf_matrix(df):
    predictions = np.array([], int)
    targets = np.array([], int)
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Processing : "):
        target = r["target"].split(' ')
        target = [int(a) for a in target]
        target = np.array(target, int)
        targets = np.concatenate((targets, target))
        prediction = r["prediction"].split(' ')
        prediction = [int(a) for a in prediction]
        prediction = np.array(prediction, int)
        predictions = np.concatenate((predictions, prediction))

    return predictions, targets

def create_confusion_matrix(csv_file, epoch=-1, frac=1):
    # raise NotImplementedError("Function is not implemented.")
    df = pd.read_csv(csv_file)
    if epoch < 0:
        # Select max epoch from csv file.
        epoch = max((df["epoch"]).unique())

    df = df.sample(frac=frac)
    df = df[df["epoch"] == epoch]
    predictions = np.array([], int)
    targets = np.array([], int)
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        target = r["target"].split(' ')
        target = [int(a) for a in target]
        target = target[1:] # Discard CLS token label.
        target = [a for a in target if a >= 0]
        target = np.array(target, int)
        targets = np.concatenate((targets, target))
        prediction = r["prediction"].split(' ')
        prediction = [int(a) for a in prediction]
        prediction = prediction[1:] # Discard token label.
        prediction = prediction[0:len(target)]
        prediction = np.array(prediction, int)
        predictions = np.concatenate((predictions, prediction))

    return predictions, targets
