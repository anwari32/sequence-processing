from math import sqrt
import os
import pandas as pd

def get_mtl_scores(log_path: str, epoch=None):
    """
    Read MTL log and count True Positive, True Negative, False Positive, and False Negative.
    """
    prom_scores, ss_scores, polya_scores = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}, {"TP": 0, "FP": 0, "TN": 0, "FN": 0}, {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    TP, FP, TN, FN = 0, 0, 0, 0
    if not os.path.exists(log_path):
        print(f"{log_path} not found.")
        raise FileNotFoundError()

    df = pd.read_csv(log_path)
    if epoch != None:
        df = df[df['epoch'] == epoch]
        
    for i, r in df.iterrows():
        prom_predict = r["prom_predict"]
        prom_label = r["prom_label"]
        ss_predict = r["ss_predict"]
        ss_label = r["ss_label"]
        polya_predict = r["polya_predict"]
        polya_label = r["polya_label"]

        if prom_predict == 1 and prom_label == 1:
            prom_scores["TP"] += 1
        elif prom_predict == 1 and prom_label == 0:
            prom_scores["FP"] += 1
        elif prom_predict == 0 and prom_label == 1:
            prom_scores["FN"] += 1
        elif prom_predict == 0 and prom_label == 0:
            prom_scores["TN"] += 1
        else:
            raise ValueError("promoter prediction: {prom_predict}, promoter label: {prom_label}")

        if ss_predict == 1 and ss_label == 1:
            ss_scores["TP"] += 1
        elif ss_predict == 1 and ss_label == 0:
            ss_scores["FP"] += 1
        elif ss_predict == 0 and ss_label == 1:
            ss_scores["FN"] += 1
        elif ss_predict == 0 and ss_label == 0:
            ss_scores["TN"] += 1
        else:
            raise ValueError("splice site prediction: {ss_predict}, splice site label: {ss_label}")

        if polya_predict == 1 and polya_label == 1:
            polya_scores["TP"] += 1
        elif polya_predict == 1 and polya_label == 0:
            polya_scores["FP"] += 1
        elif polya_predict == 0 and polya_label == 1:
            polya_scores["FN"] += 1
        elif polya_predict == 0 and polya_label == 0:
            polya_scores["TN"] += 1
        else:
            raise ValueError("polya prediction: {polya_predict}, polya label: {polya_label}")

    return prom_scores, ss_scores, polya_scores

def accuracy(tp, tn, fp, fn):
    """
    Calculates accuracy score in percent.
    """
    return (tp + tn) / (tp + tn + fp + fn) * 100

def error_rate(tp, tn, fp, fn):
    """
    Calculates error rate in percent.
    """
    return 100 - accuracy(tp, tn, fp, fn)

def specificity(tn, fp):
    """
    Calculates specificity score in percent.
    """
    return tn / (tn + fp) * 100

def sensitivity(tp, fn):
    """
    Calculates sensitivity score in percent.
    """
    return tp / (tp + fn) * 100

def precision(tp, fp):
    """
    Calculates precision in percent.
    """
    return tp / (tp + fp) * 100

def mcc(tp, tn, fp, fn):
    """
    Calculates MCC score.
    """
    return ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))