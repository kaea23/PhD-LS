# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:21:18 2025

@author: kaea23
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from discretisation import grid
from fibre_test_case import Fibre
from LS_test_case import Lippmann_Schwinger_Fibre

start = time.time()

"""
MATERIAL GEOMETRY
L: length of RVE in m
N: number of voxels x, y, z direction (N=nx,ny,nz)(2d:nz=0)
ndims: number of dimensions (2 or 3)
centre: True = centrized grid, False = non-centrized grid
scheme: S or H
"""
L = 2
N = 1000
S = 0.5
ndims = 2
centre = True
scheme = 'S'

"""
FIBRE GEOMETRY
f_num: number of fibres
f_r:fibre radius m
"""
f_num = 1 
f_r = 0.25 

"""
PARAMETERS
E_bar: average temp gradient (-ve)
epsilon: convergence criteria
c_f: fibre thermal conductivity in W/mK [2]
c_m: matrix thermal conductivity in W/mK [2]
C0: constant ( if too low can diverge)
k: kappa 1/2
"""

E_bar = np.array([2, 0], dtype = np.float64) 
epsilon = 1e-12
c_f = 2
c_m = 1
C0 = (c_f + c_m)/2 

k = c_m/c_f

E_bar = E_bar / ( 1 + ( np.pi *(1 - k * f_r**2)/(1 + k * L**2) ) )

if ndims == 2: 
    nx, ny, nz = N, N, 0 
    h1 = L/nx
    h2 = L/ny
    
    if centre:
        j1, j2 = np.meshgrid(np.arange(-nx/2, nx/2, 1),
                             np.arange(-ny/2, ny/2, 1),
                             indexing='ij')
        
    else:   
        j1, j2 = np.meshgrid(np.arange(0, nx, 1),
                             np.arange(0, ny, 1),
                             indexing='ij')

    j1 = np.moveaxis(j1, 0, 1)
    j2 = np.moveaxis(j2, 0, 1)

    x1 = (j1+S)*h1
    x2 = (j2+S)*h2
    
elif ndims == 3:
    nx, ny, nz = N, N, N
    h1 = L/nx
    h2 = L/ny
    h3 = L/nz
    
    if centre:
        j1, j2, j3 = np.meshgrid(np.arange(-nx/2, nx/2, 1),
                                 np.arange(-ny/2, ny/2, 1),
                                 np.arange(-nz/2, nz/2, 1),
                                 indexing='ij')
        
    else: 
        j1, j2, j3 = np.meshgrid(np.arange(0, nx, 1),
                                 np.arange(0, ny, 1),
                                 np.arange(0, nz, 1),
                                 indexing='ij')
        
    j1 = np.moveaxis(j1, 0, 1)
    j2 = np.moveaxis(j2, 0, 1)
    j3 = np.moveaxis(j3, 0, 1)

    x1 = (j1+S)*h1
    x2 = (j2+S)*h2
    
else: 
    raise ValueError(f"Incorrect dimensions: '{ndims}'. Supports dimensions 2 and 3 only.")
    

g = grid(nx,ny,nz,L,ndims)
xi = g.init_XI(scheme)

f_position, fy_center = Fibre(L, N, f_num, f_r)

E_n, c = Lippmann_Schwinger_Fibre(C0, L, N, xi, c_f, c_m, f_position, E_bar, epsilon, ndims, scheme)

y_index = int(fy_center[0]/h2)
x_coords = np.linspace(-L/2, L/2, N)

plt.title('FFT Gradient Temperature Profile Along y = 0')
plt.xlabel('x-coordinate')
plt.ylabel(r'$\nabla$ T')
plt.plot(x_coords, E_n[y_index, :, 0], label='$E_1(x)$')
plt.plot(x_coords, E_n[y_index, :, 1], label='$E_2(x)$')
plt.legend()
plt.grid(True)
plt.show()

plt.title('FFT $E_1(x)$')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.pcolor(x1, x2, E_n[...,0])
plt.set_cmap("nipy_spectral")
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")
plt.tight_layout()
plt.show()

plt.title('FFT $E_2(x)$')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.pcolor(x1, x2, E_n[...,1])
plt.set_cmap("nipy_spectral")
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")
plt.tight_layout()
plt.show()

end = time.time()
runtime = end-start
print(runtime)