"""
Models are based on Untari et. al. (2022) works on lung cancer mutation type and index detection.
Please refer to https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9681069 for further explanation.
"""

# Import Lib
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
from collections import Counter
import os

default_embedding_matrix = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
default_metrics = [
  'accuracy', 
  tf.keras.metrics.Precision(name="precision"),
  tf.keras.metrics.Recall(name="recall")
]

def bilstm(num_layers=2, units=256, dropout=0.2, window_size=150, num_classes=5, embedding_matrix=default_embedding_matrix, metrics=default_metrics):
    # raise NotImplementedError("This function is not implemented.")
    input = Input(shape=(window_size,)) # Input layer
    vocab_size = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    model = Embedding(vocab_size, 
                    embedding_dim, 
                    weights=[embedding_matrix],
                    input_length = window_size, 
                    trainable=False)(input)
    model = Bidirectional(LSTM(units, return_sequences=True))(model)
    if dropout>0:
        model = Dropout(dropout)(model)
    if num_layers==2:
        model = Bidirectional(LSTM(units, return_sequences=True))(model)
    if dropout>0:
        model = Dropout(dropout)(model)
    out = TimeDistributed(Dense(num_classes, activation="softmax"))(model)  # TimeDistributed wrapper layer, return sequences. Fully connected layer. 
    model = Model(input, out)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=opt, 
        metrics=metrics)
    # model.summary()
    return model


def bigru(num_layers=2, units=256, dropout=0.2, window_size=150, num_classes=5, embedding_matrix=default_embedding_matrix, metrics=default_metrics):
    # Architecture:
    # raise NotImplementedError("This function is not implemented.")
    # Architecture:
    input = Input(shape=(window_size,)) # Input layer
    vocab_size = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    model = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                            input_length = window_size, trainable=False)(input)
    model = Bidirectional(GRU(units, return_sequences=True))(model)
    if dropout>0:
        model = Dropout(dropout)(model)
    if num_layers==2:
        model = Bidirectional(GRU(units, return_sequences=True))(model)
    if dropout>0:
        model = Dropout(dropout)(model)
    out = TimeDistributed(Dense(num_classes, activation="softmax"))(model)  # TimeDistributed wrapper layer, return sequences. Fully connected layer. 
    model = Model(input, out)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=opt, 
        metrics=metrics)
    # model.summary()
    return model

if __name__ == "__main__":
    bilstm_model = bilstm()
    print(bilstm.summary())
    bigru_model = bigru()
    print(bigru.summary())

    
    
