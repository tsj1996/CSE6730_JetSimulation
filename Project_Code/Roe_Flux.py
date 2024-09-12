import numpy as np
from Flux import Flux

def Roe_Flux(U_L, U_R, N_vec):
    Ga = 1.4

    F_L = Flux(U_L, N_vec)  # Should be defined as per previous translation
    F_R = Flux(U_R, N_vec)
    v_L = np.array([U_L[1]/U_L[0],U_L[2]/U_L[0]])
    v_R = np.array([U_R[1]/U_R[0],U_R[2]/U_R[0]])
    P_L = (Ga-1.0)*(U_L[3]-0.5*((U_L[1]**2+U_L[2]**2))/U_L[0]);
    H_L = (U_L[3]/U_L[0])+P_L/U_L[0];
    P_R = (Ga-1.0)*(U_R[3]-0.5*((U_R[1]**2+U_R[2]**2))/U_R[0]);
    H_R = (U_R[3]/U_R[0])+P_R/U_R[0];
    v_Roe_avg = ((np.sqrt(U_L[0])*v_L)+(np.sqrt(U_R[0])*v_R))/(np.sqrt(U_L[0])+np.sqrt(U_R[0])); 
    H_Roe_avg = ((np.sqrt(U_L[0])*H_L)+(np.sqrt(U_R[0])*H_R))/(np.sqrt(U_L[0])+np.sqrt(U_R[0])); 
    q = np.sqrt(v_Roe_avg[0]**2+v_Roe_avg[1]**2);
    
    c = np.sqrt((Ga-1.0)*(H_Roe_avg-0.5*q**2)); 
    u_Roe = np.dot(v_Roe_avg.reshape(1,2),N_vec)
    lamda = [u_Roe+c, u_Roe-c, u_Roe];
    smax = max([abs(u_Roe)+c, abs(u_Roe)-c, abs(u_Roe)]);
    ep = 0.1*c;

    for n in range(3):  
        if abs(lamda[n]) < ep:
            lamda[n] = (ep**2 + lamda[n]**2) / (2 * ep)
    
    Fhat = 1;
    
    d_rho = U_R[0] - U_L[0]
    d_rho_v = U_R[0] * v_R - U_L[0] * v_L
    d_rho_E = U_R[3] - U_L[3]
   
    # Intermediate variables
    s1 = 0.5 * (np.abs(lamda[0]) + np.abs(lamda[1]))
    s2 = 0.5 * (np.abs(lamda[0]) - np.abs(lamda[1]))

    # G1 and G2 calculations
    G1 = (Ga - 1) * (0.5 * q**2 * d_rho - np.dot(v_Roe_avg.reshape(1,2), d_rho_v) + d_rho_E)
    G2 = -u_Roe * d_rho + np.dot(d_rho_v.reshape(1,2), N_vec)
   
    # C1 and C2 calculations
    C1 = (G1 / c**2) * (s1 - np.abs(lamda[2])) + (G2 / c) * s2
    C2 = (G1 / c) * s2 + G2 * (s1 - np.abs(lamda[2]))

    abs_lambda_3 = np.abs(lamda[2])
    Fhat_term1 = abs_lambda_3 * d_rho + C1
    Fhat_term2 = abs_lambda_3 * d_rho_v + C1 * v_Roe_avg + C2 * N_vec
    Fhat_term3 = abs_lambda_3 * d_rho_E + C1 * H_Roe_avg + C2 * u_Roe
    
    Fhat = 0.5 * (F_L.reshape(4,1) + F_R.reshape(4,1)) - 0.5 * np.vstack((Fhat_term1, Fhat_term2.reshape(2,1), Fhat_term3))
    Fhat = Fhat.reshape(1,4)
    smax = smax.reshape(1,1)
    
    return Fhat, smax



# Make sure to pass U_L, U_R as NumPy arrays with the shape (4,) and N_vec with the shape (2,)
# Input (1,4), (1,4), (1,2) output Fhat (1,4), smax (1,1)