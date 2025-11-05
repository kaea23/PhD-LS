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
#from math_comparison_test_case import mathcomp

start = time.time()

def math_comp(f_r):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Parameters
    R = f_r        # Radius of the fibre
    E0x = 2.0      # External field magnitude in x-direction
    E0y = 2.0      # External field magnitude in y-direction
    kappa = 0.5    # Conductivity ratio

    # Define Cartesian grid
    x = np.linspace(-1, 1, 500)
    y = np.linspace(-1, 1, 500)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Convert to polar coordinates
    RHO = np.sqrt(X**2 + Y**2)
    THETA = np.arctan2(Y, X)

    # Unit vectors in polar coordinates
    cos_theta = np.cos(THETA)
    sin_theta = np.sin(THETA)


    # Electric field components X direction
    Ein_phi_x = -(2 * kappa / (1 + kappa)) *E0x*RHO*cos_theta
    Eout_phi_x = (-1 + (1 - kappa) / (1 + kappa) * (R**2 / RHO**2)) *E0x*RHO*cos_theta

    # Electric field components Y direction
    Ein_phi_y = -(2 * kappa / (1 + kappa)) *E0y*RHO*sin_theta
    Eout_phi_y = (-1 + (1 - kappa) / (1 + kappa) * (R**2 / RHO**2)) *E0y*RHO*sin_theta

    # Combine based on region
    Phi_x = np.where(RHO <= R, Ein_phi_x, Eout_phi_x)
    Phi_y = np.where(RHO <= R, Ein_phi_y, Eout_phi_y)

    # Compute gradient of scalar potential (negative gradient gives electric field)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Ex,miscy = np.gradient(-Phi_x, dx, dy)
    Ey,miscx = np.gradient(-Phi_y, dx, dy)

    # Mask singularity at origin
    Ex = np.where(RHO < 0.0001, np.nan, Ex)
    Ey = np.where(RHO < 0.0001, np.nan, Ey)

    #Plot Ex component
    plt.figure(figsize=(8, 6))
    contour_ex = plt.contourf(X, Y, Ex, levels=500, cmap='seismic', vmin=1.4, vmax=2.8)
    cbar = plt.colorbar(contour_ex)
    cbar.set_label(r"$\nabla V_{E}$ ($ V $)", fontsize=14)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #circle_ex = plt.Circle((0, 0), R, color='black', fill=False, linewidth=0.2)
    #plt.gca().add_patch(circle_ex)
    plt.title('Electric Field Ex Component', fontsize=14)
    plt.xlabel('X (m)', fontsize=14)
    plt.ylabel('Y (m)', fontsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

    # Plot Ey component
    plt.figure(figsize=(8, 6))
    contour_ey = plt.contourf(X, Y, Ey, levels=500, cmap='seismic', vmin=-0.7, vmax=0.7)
    cbar = plt.colorbar(contour_ey)
    cbar.set_label(r"$\nabla V_{E}$ ($ V $)", fontsize=14)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    #circle_ey = plt.Circle((0, 0), R, color='black', fill=False, linewidth=0.2)
    #plt.gca().add_patch(circle_ey)
    plt.title('Electric Field Ey Component', fontsize=14)
    plt.xlabel('X (m)', fontsize=14)
    plt.ylabel('Y (m)', fontsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
    return Ex,Ey,kappa,R,RHO


"""
MATERIAL GEOMETRY
L: length of RVE in m
N: number of voxels x, y, z direction (N=nx,ny,nz)(2d:nz=0)
ndims: number of dimensions (2 or 3)
centre: True = centrized grid, False = non-centrized grid
scheme: S or H
"""
L = 2
N = 500
S = 0.5
ndims = 2
centre = True
scheme = 'H'

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
c_f: fibre thermal conductivity in W/mK 
c_m: matrix thermal conductivity in W/mK 
C0: constant ( if too low can diverge)
"""

E_bar = np.array([2, 0], dtype = np.float64) 
epsilon = 1e-12
c_f = 2
c_m = 1
C0 = (c_f + c_m)/2 

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
    x1 = (j1+S)*h1
    x2 = (j2+S)*h2
    
else: 
    raise ValueError(f"Incorrect dimensions: '{ndims}'. Supports dimensions 2 and 3 only.")
    

g = grid(nx,ny,nz,L,ndims)
xi = g.init_XI(scheme)

f_position, fy_center = Fibre(L, N, f_num, f_r)

E_n, c, comp_E_0 = Lippmann_Schwinger_Fibre(C0, L, N, xi, c_f, c_m, f_position, E_bar, epsilon, ndims, scheme)

Ex,Ey,kappa,R,RHO =  math_comp(f_r)

plt.figure(figsize=(8, 6))
plt.title('FFT $E_1(x)$', fontsize=14)
plt.xlabel("X (m)", fontsize=14)
plt.ylabel("Y (m)", fontsize=14)
plt.pcolor(x1, x2, E_n[...,0], vmin=1.4, vmax=2.8)
plt.set_cmap("seismic")
plt.gca().set_aspect('equal', adjustable='box')
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)", fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.title('FFT $E_2(x)$', fontsize=14)
plt.xlabel("X (m)", fontsize=14)
plt.ylabel("Y (m)", fontsize=14)
plt.pcolor(x1, x2, E_n[...,1], vmin=-0.7, vmax=0.7)
plt.set_cmap("seismic")
plt.gca().set_aspect('equal', adjustable='box')
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)", fontsize=14)
plt.tight_layout()
plt.show()

# Difference maps
diff_Ex_En0 = E_n[..., 0] - Ex
diff_Ey_En1 = E_n[..., 1] - Ey

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# First difference map
im1 = axes[0].imshow(diff_Ex_En0, cmap='seismic', extent=[-1, 1, -1, 1])
axes[0].set_title('Difference X component: FFT - Analytical solution', fontsize=14)
axes[0].set_xlabel("X (m)", fontsize=14)
axes[0].set_ylabel("Y (m)", fontsize=14)
# Second difference map
im2 = axes[1].imshow(diff_Ey_En1, cmap='seismic', extent=[-1, 1, -1, 1])
axes[1].set_title('Difference Y component: FFT - Analytical solution', fontsize=14)
axes[1].set_xlabel("X (m)", fontsize=14)
axes[1].set_ylabel("Y (m)", fontsize=14)
# Colorbars for each subplot
fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
plt.tight_layout()
plt.show()

# Line profiles
center_idx = Ex.shape[0] // 2
x = np.linspace(-1, 1, Ex.shape[1])
plt.figure(figsize=(8, 6))
plt.plot(x, Ex[center_idx, :], label='Ex (Analytical)')
plt.plot(x, E_n[center_idx, :, 0], label='$E_1(x)$ (FFT)')
plt.legend()
plt.title('Line Profile: X component', fontsize=14)
plt.xlabel("X (m)", fontsize=14)
plt.ylabel(r"gradient $\nabla$", fontsize=14)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x, Ey[center_idx, :], label='Ey (Analytical)')
plt.plot(x, E_n[center_idx, :, 1], label='$E_2(x)$(FFT)')
plt.legend()
plt.title('Line Profile: Y component', fontsize=14)
plt.xlabel("X (m)", fontsize=14)
plt.ylabel(r"gradient $\nabla$", fontsize=14)
plt.show()

# Correction term
#correction_term = ((1 - kappa) / (1 + kappa)) * (R**2 / RHO**2)
#correction_term = np.where(RHO < 0.0001, np.nan, correction_term)

end = time.time()
runtime = end-start
print(runtime)

