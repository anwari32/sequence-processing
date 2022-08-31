from seqlab import NUM_LABELS, Index_Dictionary, Label_Dictionary
import numpy as np
import torch

def accuracy_and_error_rate(input_ids, prediction, target):
    if not (input_ids.shape == prediction.shape == target.shape):
        raise ValueError(f"All inputs must have 1 dimension. Found {input_ids.shape}, {prediction.shape}, {target.shape}")

    # Remove CLS token
    input_ids = input_ids[1:]
    prediction = prediction[1:]
    target = target[1:]

    # Remove special tokens.
    input_ids = [a for a in input_ids if a >= 0]
    input_ids_len = len(input_ids)
    prediction = prediction[0:input_ids_len]
    target = target[0:input_ids_len]

    accuracy = 0
    for i, j in zip(prediction, target):
        accuracy = accuracy + (1 if i == j else 0)
    accuracy = accuracy / input_ids_len
    return accuracy, (1 - accuracy)

class Metrics:
    r"""
    Make sure to use CLEAN prediction and target indices;
    prediction and target indices must contain ONLY labels.
    """
    def __init__(self, prediction, target):
        r"""
        * :attr:`prediction`
        * :attr:`target`
        """
        if isinstance(prediction, torch.Tensor):
            raise TypeError(f"prediction type error. expected type Array found {type(prediction)}")
        if isinstance(target, torch.Tensor):
            raise TypeError(f"target type error. expected type Array found {type(target)}")

        self.prediction = prediction
        self.matrix = []
        for i in range(NUM_LABELS):
            _m = []
            for j in range(NUM_LABELS):
                _m.append(0)    
            self.matrix.append(_m)

        self.target = target
        self.labels = Label_Dictionary.keys()
        self.indices = [k for k in range(8)]
        self.Trues = {}
        self.Falses = {}
        for k in self.labels:
            self.Trues[k] = 0
            self.Falses[k] = 0 
        self.special_tokens = 0
    
    def get_label_counts(self):
        return {
            "Trues": self.Trues,
            "Falses": self.Falses
        } 

    def calculate(self):
        n_pred = len(self.prediction)
        n_target = len(self.target)
        if n_pred != n_target:
            raise ValueError(f"Prediction and target are not the same size. Found {n_pred} and {n_target}")
        for p, t in zip(self.prediction, self.target):
            pred_label =  Index_Dictionary[p]
            target_label = Index_Dictionary[t]
            self.matrix[p][t] += 1

            if pred_label == "[CLS]/[SEP]/[III]":
                self.special_tokens += 1
            elif pred_label == target_label:
                self.Trues[pred_label] += 1
            else:
                self.Falses[pred_label] += 1

    def get_precision(self, percentage=False):
        dummy = {}
        for label_index in range(8):
            precision = self.precission(label_index, percentage=percentage)
            dummy[label_index] = precision
        print(dummy)
        return dummy

    def precission(self, label_index, percentage=False):
        label = Index_Dictionary[label_index]
        ret = 0
        try:
            ret = self.Trues[label] / (self.Falses[label] + self.Trues[label])
        except ZeroDivisionError:
            ret = 0 # Set to zero if things went south.
        return ret * (100 if percentage else 1)

    def recall(self, label_index, percentage=False):
        label = Index_Dictionary[label_index]
        rows = self.matrix[label_index]
        sum = np.sum([rows[i] for i in range(NUM_LABELS) if i != label_index])
        ret = 0
        try:
            ret = self.Trues[label] / (self.Trues[label] + sum)
        except ZeroDivisionError:
            ret = 0 # Set to zero if things went south.
        return ret * (100 if percentage else 1)
        
        
if __name__ == "__main__":
    rand_prediction = np.random.randint(0, 3, size=10)
    rand_target = np.random.randint(0, 3, size=10)
    metrics = Metrics(rand_prediction, rand_target)
    metrics.calculate()
    print(f"prediction {rand_prediction}")
    print(f"target {rand_target}")
    print("Precision")
    for k in metrics.indices:
        print(f"{k} => {metrics.precission(k, True)}")
    print("Recall")
    for k in metrics.indices:
        print(f"{k} => {metrics.recall(k, True)}")



