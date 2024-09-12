import numpy as np

def get_N_vec(A, B):
    x_A, y_A = A
    x_B, y_B = B
    dl = np.sqrt((x_A - x_B)**2 + (y_A - y_B)**2)
    N_vec = np.array([y_B - y_A, x_A - x_B]) / dl
    return N_vec

