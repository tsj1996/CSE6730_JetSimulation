# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:44:15 2023

@author: tsj19
"""
# Nn: number of nodes
# Ne: number of elements
# dim: dimension
# V: vertices
# E: elements
# IE: internal edge
# BE: boundary edge
# B: boundary nodes


import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from readgrd import readgrd
from get_N_vec import get_N_vec
from get_dl import get_dl
from Roe_Flux import Roe_Flux
from get_pt import get_pt
from get_Mach import get_Mach


Mesh = readgrd('mesh0.grd')
#Mesh = readgrd('a3_mesh5.grd')

#[IE, BE] = edgehash(Mesh['E'], Mesh['B'])

Gamma = 1.4;    #specific heat ratio
a = 1*math.pi/180;   #angle of attack   
M_inf = 2.2;    #free stream Mach number
n_ite = 1200;   # number of iterations
CFL = 1;    # CFL number

#Initialization
U = np.zeros((Mesh['Ne'],4)) #vector containing all the parameters
U_inf = np.zeros((Mesh['Ne'],4)) #initial vector
dt_Ai = np.zeros((Mesh['Ne'],4)) #
R_L1_tot = np.zeros((n_ite,1))
ATPR = np.zeros((n_ite,1))

#Initial Condition
U_free = np.array([1, M_inf*math.cos(a), M_inf*math.sin(a),(1/((Gamma-1)*Gamma))+(M_inf**2)/2])

"""
Created on Sun Nov 19 09:58:48 2023

@author: kaiqu
"""


for i in range(Mesh['Ne']):
    U[i,:] = U_free
    U_inf[i,:] = U_free


for i in range(n_ite):
    R = np.zeros((Mesh['Ne'],4))
    s_dl = np.zeros((Mesh['Ne'],1))
    R_L1_tot[i] = 0;
    v_plus = np.zeros((1,2))
    Temp_Fhat = np.zeros((1,2))
    
    for j in range(len(Mesh['BE'])):
        BE_Elem_Index = Mesh['BE'][j,2]
        BE_Elem_NVect = get_N_vec(Mesh['V'][Mesh['BE'][j,0], :], Mesh['V'][Mesh['BE'][j,1], :])        

        if Mesh['BE'][j,3] == 0:  #Engine, wall

            v_plus = np.array([U[BE_Elem_Index, 1]/U[BE_Elem_Index, 0], U[BE_Elem_Index, 2]/U[BE_Elem_Index, 0]])
            v_n = np.dot(np.dot(v_plus,BE_Elem_NVect), BE_Elem_NVect)
            v_b = v_plus - v_n
            
            p_b = (Gamma - 1)*(U[BE_Elem_Index, 3] - 0.5*U[BE_Elem_Index, 0]*(np.linalg.norm(v_b)**2))
            p = (Gamma - 1)*(U[BE_Elem_Index, 3] - 0.5*U[BE_Elem_Index, 0]*(np.linalg.norm(v_plus)**2))
            #print(p)
            
            c = np.sqrt((Gamma - 1)*((U[BE_Elem_Index, 3]/U[BE_Elem_Index,0]) - 0.5*(np.linalg.norm(v_plus)**2)))
            #print(c)
            Temp_Fhat = np.dot(p_b, BE_Elem_NVect)
            
            Fhat = np.array([0, Temp_Fhat[0], Temp_Fhat[1], 0])
            #print(Fhat)
            smax = np.linalg.norm(v_n) + c
            dl = get_dl(Mesh['V'][Mesh['BE'][j,0], :], Mesh['V'][Mesh['BE'][j,1], :])
            R[BE_Elem_Index, :] = R[BE_Elem_Index, :] + dl*Fhat
            s_dl[BE_Elem_Index, :] = s_dl[BE_Elem_Index, :] + smax*dl
            #print(Fhat)
        
        elif Mesh['BE'][j,3] == 3: #inflow
            
            [Fhat, smax] = Roe_Flux(U[BE_Elem_Index, :], U_inf[0,:], BE_Elem_NVect)
            dl = get_dl(Mesh['V'][Mesh['BE'][j,0], :], Mesh['V'][Mesh['BE'][j,1], :])
            R[BE_Elem_Index, :] = R[BE_Elem_Index, :] + dl*Fhat
            s_dl[BE_Elem_Index, :] = s_dl[BE_Elem_Index, :] + np.abs(smax)*dl  
            
        else:
            
            [Fhat, smax] = Roe_Flux(U[BE_Elem_Index, :], U[BE_Elem_Index, :], BE_Elem_NVect)
            dl = get_dl(Mesh['V'][Mesh['BE'][j,0], :], Mesh['V'][Mesh['BE'][j,1], :])
            R[BE_Elem_Index, :] = R[BE_Elem_Index, :] + dl*Fhat
            s_dl[BE_Elem_Index, :] = s_dl[BE_Elem_Index, :] + smax*dl    

    for j in range(len(Mesh['IE'])):
        
        IE_Elem1_Index = Mesh['IE'][j,2]
        IE_Elem2_Index = Mesh['IE'][j,3]
        IE_Elem_NVect = get_N_vec(Mesh['V'][Mesh['IE'][j,0], :], Mesh['V'][Mesh['IE'][j,1], :])
        
        [Fhat, smax] = Roe_Flux(U[IE_Elem1_Index, :], U[IE_Elem2_Index, :], IE_Elem_NVect)
        dl = get_dl(Mesh['V'][Mesh['IE'][j,0], :], Mesh['V'][Mesh['IE'][j,1], :])
        R[IE_Elem1_Index, :] = R[IE_Elem1_Index, :] + dl*Fhat
        s_dl[IE_Elem1_Index, :] = s_dl[IE_Elem1_Index, :] + np.abs(smax)*dl           
        
        [Fhat, smax] = Roe_Flux(U[IE_Elem2_Index, :], U[IE_Elem1_Index, :], -IE_Elem_NVect)
        dl = get_dl(Mesh['V'][Mesh['IE'][j,0], :], Mesh['V'][Mesh['IE'][j,1], :])
        R[IE_Elem2_Index, :] = R[IE_Elem2_Index, :] + dl*Fhat
        s_dl[IE_Elem2_Index, :] = s_dl[IE_Elem2_Index, :] + np.abs(smax)*dl  

    for j in range(Mesh['Ne']):
        dt_Ai[j] = 2*CFL/s_dl[j]
        U[j,:] = U[j,:] - dt_Ai[j]*R[j,:]
        Rj = np.abs(R[j,0]) + np.abs(R[j,1]) + np.abs(R[j,2]) + np.abs(R[j,3])
        R_L1_tot[i] = R_L1_tot[i] + Rj
        
    
    ##ATPR
    pt_free = get_pt(U_free)
    d = 1
    pt_intg = np.array([])
    
    
    for j in range(len(Mesh['BE'])):
        if Mesh['BE'][j,3] == 1:
           BE_Elem_Index = Mesh['BE'][j,2]
           dl = get_dl(Mesh['V'][Mesh['BE'][j,0], :], Mesh['V'][Mesh['BE'][j,1], :])
           pt = get_pt(U[BE_Elem_Index, :])
           pt_intg = np.append(pt_intg, (pt/pt_free)*dl)
    
    ATPR[i] = (1/d)*np.sum(pt_intg) 
    print(i)
 
 
# Create the first plot
plt.figure(1, dpi=300)
plt.plot(R_L1_tot, linewidth=2.5)  # Plotting the data
plt.xlabel('Iterations')
plt.ylabel('Residual L1 Norm')
plt.title('Residual L1 Norm Convergence')
plt.gca().lineWidth = 2.5
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

# Create the second plot
plt.figure(2)
plt.plot(ATPR, linewidth=2.5)  # Plotting the data
plt.xlabel('Iterations')
plt.ylabel('ATPR')
plt.title('ATPR Convergence') 
plt.gca().lineWidth = 2.5
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
 
points = []
faces = []
color1 = []
color2 = []
k = 0

for i in range(Mesh['Ne']):
    face = []
    for j in range(3):
        point_index = Mesh['E'][i, j]
        point = Mesh['V'][point_index, :2]
        points.append(point)
        face.append(k)
        color1.append(get_Mach(U[i, :]))
        color2.append(get_pt(U[i, :]))
        k += 1
    faces.append(face)

points = np.array(points)
faces = np.array(faces)
color1 = np.array(color1)
color2 = np.array(color2)

# Mach Number Field Plot
plt.figure(3, dpi=300)
triang = mtri.Triangulation(points[:, 0], points[:, 1], faces)
plt.tripcolor(triang, color1, edgecolors='white', cmap = 'jet')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mach Number Field Plot')
plt.gca().lineWidth = 2
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Total Pressure Field Plot
plt.figure(4, dpi=300)
triang = mtri.Triangulation(points[:, 0], points[:, 1], faces)
plt.tripcolor(triang, color2, edgecolors='white', cmap = 'jet')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Total Pressure Field Plot')
plt.gca().lineWidth = 2
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()