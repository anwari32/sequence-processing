# DUMP ALL SCRIPTS!

from tf_model.wisesty import bigru, bilstm
import numpy as np
import pandas as pd
import pickle
import time
import keras
import keras.utils
import tensorflow as tf
from keras.layers import Embedding, Dense, Flatten, Dropout, SpatialDropout1D, TimeDistributed, LSTM, GRU, Bidirectional
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from keras import Input, Model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
import os

nucleotide_dict = {
    "T": 1,
    "C": 2,
    "A": 3,
    "G": 4, 
    "N": 0
}

exon_intron_dict = {
    "i": 0,
    "E": 1,
    "N": -100,
}

num_epochs = 20
batch_size = 48
num_classes = 3

def compute_f1_score(precision, recall):
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def convert_label(y, num_classes):
    if y in [0, 1]:
        return tf.keras.utils.to_categorical(y, num_classes)
    else:
        return [0, 0]


def preprocessing(data_path, num_classes=2):
    encoded_sequences = []
    encoded_labels = []
    df = pd.read_csv(data_path)
    for i, r in df.iterrows():
        sequence = r["sequence"]
        label = r["label"]

        # padding sequence.
        encoded_sequence = [nucleotide_dict[a] for a in list(sequence)]
        if len(encoded_sequence) < 150:
            delta = 150 - len(encoded_sequence)
            for j in range(delta):
                encoded_sequence.append(0)

        # padding label.
        encoded_label = [exon_intron_dict[a] for a in list(label)]
        if len(encoded_label) < 150:
            delta = 150 - len(encoded_label)
            for j in range(delta):
                encoded_label.append(-100)

        encoded_sequences.append(
            encoded_sequence
        )
        encoded_labels.append(
            encoded_label
        )

    encoded_sequences = np.array(encoded_sequences)
    encoded_labels = np.array([[convert_label(_y, num_classes) for _y in y] for y in encoded_labels])
    return encoded_sequences, encoded_labels


if __name__ == "__main__":

    bigru_model = bigru(num_classes=num_classes)
    bilstm_model = bilstm(num_classes=num_classes)
    
    work_dir = os.path.join("workspace", "baseline", "basic")
    training_data_path = os.path.join(work_dir, "gene_index.01_train_validation_ss_all_pos_train.csv")
    validation_data_path = os.path.join(work_dir, "gene_index.01_train_validation_ss_all_pos_validation.csv")
    test_data_path = os.path.join(work_dir, "gene_index.01_test_ss_all_pos.csv")

    X_train, Y_train = preprocessing(training_data_path)
    X_val, Y_val = preprocessing(validation_data_path)
    X_test, Y_test = preprocessing(test_data_path)

    print(f"Training data {np.array(X_train).shape}, {all([len(v) == 150 for v in X_train])}, {np.array(Y_train).shape}, {all([len(v) == 150 for v in Y_train])}")
    print(f"Validation data {np.array(X_val).shape}, {all([len(v) == 150 for v in X_val])}, {np.array(Y_val).shape}, {all([len(v) == 150 for v in Y_val])}")
    print(f"Test data {np.array(X_test).shape}, {all([len(v) == 150 for v in X_test])}, {np.array(Y_test).shape}, {all([len(v) == 150 for v in Y_test])}")

    run_dir = os.path.join("run", "baseline", "basic")
    log_dir = os.path.join(run_dir, "log")
    model_dir = os.path.join(run_dir, "model")

    for p in ["model", "log"]:
        d = os.path.join(run_dir, p)
        os.makedirs(p, exist_ok=True)

    for m in [("bilstm", bilstm_model), ("bigru", bigru_model)]:
        model_name = m[0]
        model = m[1]

        train_history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=num_epochs, batch_size=batch_size)
        model.save(
            os.path.join(model_dir, f"model_{model_name}.h5")
        )
        hist_keys = train_history.history.keys()
        hist_data = {}
        for k in hist_keys:
            hist_data[k] = train_history.history[k]

        train_f1_score = []
        val_f1_score = []

        for p, r in zip(train_history.history["precision"], train_history.history["recall"]):
            train_f1_score.append(
                compute_f1_score(p, r)
            )

        for p, r in zip(train_history.history["val_precision"], train_history.history["val_recall"]):
            val_f1_score.append(
                compute_f1_score(p, r)
            )

        hist_data["f1_score"] = train_f1_score
        hist_data["val_f1_score"] = val_f1_score

        training_validation_result_df = pd.DataFrame(data=hist_data)
        training_validation_result_df.to_csv(
            os.path.join(log_dir, f"log.{model_name}.csv"), 
            index=False)


