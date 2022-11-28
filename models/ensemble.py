# ensemble implementation

import torch
from utils.seqlab import Index_Dictionary

class Ensemble():
    def __init__(self, pred_bilstm=None, pred_bigru=None, pred_dnabert=None):
        self.bilstm_prediction = pred_bilstm
        self.bigru_prediction = pred_bigru
        self.dnabert_prediction = pred_dnabert
        self.final_prediction = []
        self.label_vote = {}
        for k in Index_Dictionary.keys():
            if k >= 0:
                self.label_vote[k] = 0

    def consolidate(self):
        for i in range(len(self.bilstm_prediction)):
            bilstm_vote = self.bilstm_prediction[i]
            bigru_vote = self.bigru_prediction[i]
            dnabert_vote = self.dnabert_prediction[i]

