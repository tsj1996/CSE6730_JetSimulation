import numpy as np

def get_Mach(U):
    Ga = 1.4
    v = np.array([U[1] / U[0], U[2] / U[0]])  # Use numpy array for vector operations
    p = (Ga - 1) * (U[3] - 0.5 * U[0] * np.linalg.norm(v)**2)
    c = np.sqrt(Ga * p / U[0])
    M = np.linalg.norm(v) / c
    return M

