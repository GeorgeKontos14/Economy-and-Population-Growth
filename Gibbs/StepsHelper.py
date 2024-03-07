import numpy as np

def find_B(A, C):
    a = len(A)
    c = len(C)
    B = np.zeros((a, c))
    for i in range(a):
        B[i, :] = A[i]*C
    return B.T