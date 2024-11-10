import numpy as np
import math

def running_average_smoothing(y, n_samples=10):
    y_smooth = []
    for i in range(0, y.shape[0]):
        avg = 0
        if i < (n_samples-1):
            avg = np.average(y[0:i+1])
        else:
            avg = np.average(y[i-(n_samples-1):i+1])
            if math.isnan(avg):
                raise ValueError(f"index {i}, values {y[i-(n_samples-1):i+1]}")
        y_smooth.append(avg)
    return y_smooth