import numpy as np

def get_pt(U):
    Ga = 1.4
    v = np.array([U[1] / U[0], U[2] / U[0]])  # note the index shift for Python
    p = (Ga - 1) * (U[3] - 0.5 * U[0] * np.linalg.norm(v)**2)
    #print(p)
    #print(U[0])
    
    c = np.sqrt(Ga * p / U[0])
    M = np.linalg.norm(v) / c
    pt = p * (1 + (Ga - 1) * 0.5 * M**2)**(Ga / (Ga - 1))
    return pt

