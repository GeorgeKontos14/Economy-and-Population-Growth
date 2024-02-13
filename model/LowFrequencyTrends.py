import numpy as np

def compute_y(f, c):
    y = np.copy(c)
    for i in range(y.shape[0]):
        y[i] += f
    return y