# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:28:25 2025

@author: kaea23
"""

import numpy as np

# For single fibre in centre of domain

def Fibre(L, N, f_num, f_r):
    h1, h2 = L/N, L/N
    
    A_mat = L*L
    A_f = f_num*np.pi*(f_r)**2 
    f_percent = (A_f/A_mat)*100
    
    fx_center = [L/2]
    fy_center = [L/2]
    
    print('x centre',fx_center) 
    print('y centre',fy_center) 
    
    f_position = np.full((N,N), False, dtype=bool)
       
    x, y = np.meshgrid(np.arange(0,N), np.arange(0,N))
    for i in range(f_num):
        idx = (x*h1-fx_center[i])**2 + (y*h2-fy_center[i])**2 <= f_r**2
        f_position[idx] = 1
    
    print('  Fibre volume percentage %.2f  %%'%(f_percent))
    return f_position, fy_center

