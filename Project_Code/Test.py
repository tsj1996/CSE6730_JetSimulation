# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:55:19 2023

@author: tsj19
"""

import numpy as np
from Roe_Flux import Roe_Flux

U_L = np.array([1, 2.2, 0, 4.2057])
U_R = np.array([1, 2.5, 0, 4.9107])
n = np.array([1,2])

[Fhat, smax] = Roe_Flux(U_L, U_R, n)

print(Fhat)
print(smax)