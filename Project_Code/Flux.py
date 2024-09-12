import numpy as np

def Flux(U, N):
    Ga = 1.4
    
    F = np.zeros((4, 2))
    P = (Ga - 1) * (U[3] - 0.5 * ((U[1]**2 + U[2]**2) / U[0]))
    H = (U[3] / U[0]) + P / U[0]
    
    F[0, 0] = U[1]
    F[0, 1] = U[2]
    F[1, 0] = (U[1]**2 / U[0]) + P
    F[1, 1] = U[1] * U[2] / U[0]
    F[2, 0] = U[1] * U[2] / U[0]
    F[2, 1] = (U[2]**2 / U[0]) + P
    F[3, 0] = U[1] * H
    F[3, 1] = U[2] * H
    F = np.dot(F, N)
    
  
    return F
