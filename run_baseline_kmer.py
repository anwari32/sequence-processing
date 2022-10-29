# DUMP ALL SCRIPTS!

from models.wisesty import bigru, bilstm
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
from utils.seqlab import token2id
import numpy as np
import pandas as pd
from tqdm import tqdm


num_epochs = 20
batch_size = 48
num_classes = 8
window_size = 510

def compute_f1_score(precision, recall):
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

def convert_label(y, num_classes):
    # if y in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    #     return tf.keras.utils.to_categorical(y, num_classes)
    # else:
    #     return [0, 0, 0, 0, 0, 0, 0, 0]
    vector = [0 for i in range(num_classes)]
    if y in [i for i in range(num_classes)]:
        vector[y] = 1
    return vector

def __kmer_embedding__():
    nucleotides = ["A", "C", "G", "T"]
    token_id = 0
    embedding = []
    embedding.append([0 for i in range(64)]) # added padding encoding
    kmer_dict = {}
    kmer_dict["NNN"] = token_id
    token_id = 1
    for a in nucleotides:
        for b in nucleotides:
            for c in nucleotides:
                token = f"{a}{b}{c}"
                token_vector = [0 for i in range(64)]
                token_vector[(token_id - 1)] = 1
                embedding.append(token_vector)
                kmer_dict[token] = token_id
                token_id += 1

    return np.array(embedding), kmer_dict

default_kmer_embedding_matrix, default_kmer_dictionary = __kmer_embedding__()

def preprocessing(data_path, vocabulary=default_kmer_dictionary, window_size=510):
    encoded_sequences = []
    encoded_labels = []
    df = pd.read_csv(data_path)
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Preprocessing"):
        sequence = r["sequence"].split(" ")
        label = r["label"].split(" ")

        # padding sequence.
        encoded_sequence = [vocabulary[a] for a in sequence]
        if len(encoded_sequence) < window_size:
            delta = window_size - len(encoded_sequence)
            for j in range(delta):
                encoded_sequence.append(0)

        # padding label.
        encoded_label = [token2id(a) for a in label]
        if len(encoded_label) < window_size:
            delta = window_size - len(encoded_label)
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

    kmer_embedding, kmer_dict = __kmer_embedding__()
    print(kmer_embedding, kmer_embedding.shape)
    print(kmer_dict)

    bigru_model = bigru(window_size=510, num_classes=num_classes, embedding_matrix=default_kmer_embedding_matrix)
    bilstm_model = bilstm(window_size=510, num_classes=num_classes, embedding_matrix=default_kmer_embedding_matrix)
    
    work_dir = os.path.join("workspace", "seqlab-latest")
    training_data_path = os.path.join(work_dir, "gene_index.01_train_validation_ss_all_pos_train.csv")
    validation_data_path = os.path.join(work_dir, "gene_index.01_train_validation_ss_all_pos_validation.csv")
    test_data_path = os.path.join(work_dir, "gene_index.01_test_ss_all_pos.csv")

    X_train, Y_train = preprocessing(training_data_path)
    X_val, Y_val = preprocessing(validation_data_path)
    X_test, Y_test = preprocessing(test_data_path)

    print(f"Training data {np.array(X_train).shape}, {all([len(v) == window_size for v in X_train])}, {np.array(Y_train).shape}, {all([len(v) == window_size for v in Y_train])}")
    print(f"Validation data {np.array(X_val).shape}, {all([len(v) == window_size for v in X_val])}, {np.array(Y_val).shape}, {all([len(v) == window_size for v in Y_val])}")
    print(f"Test data {np.array(X_test).shape}, {all([len(v) == window_size for v in X_test])}, {np.array(Y_test).shape}, {all([len(v) == window_size for v in Y_test])}")

    run_dir = os.path.join("run", "baseline", "kmer")
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