from multiprocessing.sharedctypes import Value
from .seqlab import NUM_LABELS, Index_Dictionary, Label_Dictionary
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

def cleanup_prediction(prediction, target):
    _psize = np.asarray(prediction).size
    _tsize = np.asarray(target).size
    if _psize != _tsize:
        raise ValueError(f"prediction and target are not in same size. Found {_psize} and {_tsize}")
    ct = target[1:] # Remove CLS token at first index.
    ct = [a for a in ct if a >= 0] # Remove special tokens indicated by negative number.
    cp = prediction[1:] # Remove CLS token at first index.
    cp = cp[0:len(ct)] # Remove any prediction on special tokens by matching the size of cp tp ct.
    
    return cp, ct

def clean_prediction_target_batch(prediction, target):
    cp, ct = [], [] # cp: clean prediction, ct: clean target.
    for p, t in zip(prediction, target):
        if t != -100:
            cp.append(p)
            ct.append(t)
    return cp, ct


metric_names = ["precision", "recall", "f1_score"]

class Metrics:
    """
    Make sure to use CLEAN prediction and target indices;
    prediction and target indices must contain ONLY labels.
    """
    def __init__(self, prediction=[], target=[], num_classes=8, labels=[]):
        r"""
        * :attr:`prediction`    (Array | None -> None)
        * :attr:`target`        (Array | None -> None)
        * :attr:`num_classes`   (Array | None -> None)
        """
        self.num_classes = num_classes
        self.labels = labels
        self.possible_label_ids = [i for i in range(self.num_classes)]
        if len(prediction) > 0 and len(target) > 0:
            if isinstance(prediction, torch.Tensor):
                raise TypeError(f"prediction type error. expected type Array found {type(prediction)}")
            if isinstance(target, torch.Tensor):
                raise TypeError(f"target type error. expected type Array found {type(target)}")

            self.prediction = [int(a) for a in prediction]
            self.target = [int(a) for a in target]
        else:
            self.prediction = []
            self.target = []

        
        # matrix. horizontal represents 'prediction', vertical represents 'target'.
        # i.e. matrix[x][y]: how many items are predicted as x but are, in fact, y in reality.
        self.matrix = []
        for i in range(self.num_classes):
            _m = [0 for j in range(self.num_classes)]
            self.matrix.append(_m)

        self.target = [int(a) for a in target]
        self.indices = [k for k in range(self.num_classes)]
        self.Trues = {}
        self.Falses = {}
        for k in self.labels:
            self.Trues[k] = 0
            self.Falses[k] = 0 
        self.special_tokens = 0

    def print_cf(self):
        for i in range(len(self.matrix)):
            print(np.asarray([self.matrix[p][i] for p in range(self.num_classes)]))
    
    def get_label_counts(self):
        return {
            "Trues": self.Trues,
            "Falses": self.Falses
        } 

    def calculate(self, y_prediction=[], y_target=[]):
        n_pred = len(self.prediction) if len(y_prediction) == 0 else len(y_prediction)
        n_target = len(self.target) if len(y_target) == 0 else len(y_target)
        self.prediction.extend(y_prediction)
        self.target.extend(y_target)
        if n_pred != n_target:
            raise ValueError(f"Prediction and target are not the same size. Found {n_pred} and {n_target}")
        for p, t in zip(self.prediction, self.target):
            if p not in self.possible_label_ids:
                raise IndexError(f"index {p} out of bounds")
            if t not in self.possible_label_ids:
                raise IndexError(f"index {t} out of bounds")
            self.matrix[p][t] += 1

    def extend(self, y_pred, y_target):
        self.prediction.extend(y_pred)
        self.target.extend(y_target)
        for p, t in zip(y_pred, y_target):
            self.matrix[p][t] += 1

    def true_label(self, label_index):
        r"""
        Return number of occurences where predicted label is indeed target label.
        """
        if label_index >= self.num_classes or label_index < 0:
            raise IndexError(f"label index out of bounds {label_index}")
        return self.matrix[label_index][label_index]

    def false_label(self, label_index):
        r"""
        Return number of occurences where predicted label is not target label hence false label.
        """
        if label_index >= self.num_classes or label_index < 0:
            raise IndexError(f"label index out of bounds {label_index}")
        return sum([self.matrix[label_index][a] for a in range(self.num_classes) if a != label_index])

    def false_non_label(self, label_index):
        r"""
        Return number of occurences where target label is not predicted label hence false non label.
        """
        if label_index >= self.num_classes or label_index < 0:
            raise IndexError(f"label index out of bounds {label_index}")
        return sum([self.matrix[a][label_index] for a in range(self.num_classes) if a != label_index])

    def precision(self, label_index, percentage=False):
        ret = 0
        try:
            t_label = self.true_label(label_index)
            f_label = self.false_label(label_index)
            ret = t_label / (t_label + f_label)
        except ZeroDivisionError:
            ret = 0 # Set to zero if things went south.
        return round(ret * (100 if percentage else 1), 3)

    def recall(self, label_index, percentage=False):
        ret = 0
        try:
            t_label = self.true_label(label_index)
            f_non_label = self.false_non_label(label_index)
            ret = t_label / (t_label + f_non_label)
        except ZeroDivisionError:
            ret = 0 # Set to zero if things went south.
        return round(ret * (100 if percentage else 1), 3)

    def f_score(self, label_index, beta):
        ret = 0
        try:
            precision = self.precision(label_index)
            recall = self.recall(label_index)
            ret = (1 + (beta*beta)) * (precision * recall) / ((beta*precision) + recall)
        except ZeroDivisionError:
            ret = 0
        return ret

    def f1_score(self, label_index):
        return round(self.f_score(label_index, 1), 3)

    def accuracy_and_error_rate(self):
        if len(self.prediction) != len(self.target):
            raise ValueError(f"size mismatch. found {len(self.prediction)} and {len(self.target)}")
        accuracy = 0
        for i, j in zip(self.prediction, self.target):
            accuracy = accuracy + (1 if i == j else 0)
        accuracy = accuracy / len(self.prediction)
        return accuracy, (1 - accuracy)
        
        
if __name__ == "__main__":
    rand_prediction = [5, 2, 4, 2, 7, 1, 4, 3, 2, 6]
    rand_target     = [3, 7, 7, 7, 1, 2, 7, 1, 2, 3]
    print(f"Prediction  {np.asarray(rand_prediction)}")
    print(f"Target      {np.asarray(rand_target)}")
    metrics = Metrics(rand_prediction, rand_target)
    metrics.calculate()
    metrics.print_cf()
    for i in range(8):
        print(f"Precission  [{i}]{metrics.precission(i)}")
        print(f"Recall      [{i}]{metrics.recall(i)}")

