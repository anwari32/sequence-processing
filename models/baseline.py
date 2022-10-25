"""
Baseline model is based on Untari et. al. (2022) works on lung cancer mutation type and index detection.
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



