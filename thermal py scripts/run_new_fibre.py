# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:10:25 2025

@author: kaea23
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from discretisation import grid
from distributions_fibres import FibreRadiusDistribution, FibreDistribution2d
from LS_Fibre_Test import Lippmann_Schwinger_Fibre_Gauss

start = time.time()

"""
MATERIAL GEOMETRY
L: length of RVE in m
N: number of voxels x, y, z direction (N=nx,ny,nz)(2d:nz=0)
ndims: number of dimensions (2 or 3)
S: centre offset of grid
"""

L = 50.0e-6
N = 500
S = 0.5
ndims = 2

"""
FIBRE GEOMETRY

[1] Potential strength of Nicalone, Hi Nicalone, and Hi Nicalon Type S
monofilaments of variable diameters : Hi-Nicalon Type S

r_avg: fibre radius average micro m [1]
r_min: fibre radius min micro m [1]
r_max: fibre radius max micro m [1]
sigma: fibre radius standard deviation micro m SD [1]
gaussian: True/False
"""

r_avg = 7.65e-6 
r_min = 4.7e-6 
r_max = 9.75e-6 
sigma = 1.6e-6 
gaussian = True

FRD = FibreRadiusDistribution(r_avg, r_min, r_max, sigma, gaussian)

"""
PARAMETERS
E_bar: average temp gradient (-ve)
epsilon: convergence criteria
c_f: fibre thermal conductivity in W/mK [2]
c_m: matrix thermal conductivity in W/mK [2]
C0: constant ( if too low can diverge)
rho_c_m: density matrix cubic SiC in kg/m^3 [https://www.qualitymaterial.net/news_list85.html]
rho_c_f: density fibre Hi-Nicalon Type S kg/m^3 [1]
cp: heat capacity in J m^-3 K^-1  [2]
domain_size: L
volume_fraction: volume fraction of fibres
kdiff_background: diffusion coefficient in background D = k/(ρ*cp)
kdiff_fibre: diffusion coefficient in fibre D = k/(ρ*cp)
r_fibre_dist: fibre radius distribution, instance of class FibreRadiusDistribution
fast_code: True/False
"""

E_bar = np.array([1, 1], dtype = np.float64) 
epsilon = 1e-8
c_f = 22 
c_m = 120 
C0 = c_f + c_m/2 
rho_c_m = 3210 
rho_c_f = 3100 
cp = 2210000 
domain_size = L
volume_fraction = 0.55
kdiff_background = c_m/(rho_c_m * cp) 
kdiff_fibre = c_f/(rho_c_f * cp) 
r_fibre_dist = FRD
fast_code = True


FD2D = FibreDistribution2d(N, domain_size, r_fibre_dist, volume_fraction, 
                           kdiff_background, kdiff_fibre, fast_code)

fibre_location = FD2D.__iter__()
f_position = next(fibre_location)

if ndims == 2:
    nx, ny, nz = N+1, N+1, 0 
    h1 = L/nx
    h2 = L/ny
    
    j1, j2 = np.meshgrid(np.arange(0, nx, 1),
                         np.arange(0, ny, 1))

    j1 = np.moveaxis(j1, 0, 1)
    j2 = np.moveaxis(j2, 0, 1)

    x1 = (j1+S)*h1
    x2 = (j2+S)*h2
else:
    nx, ny, nz = N+1, N+1, N+1
    h1 = L/nx
    h2 = L/ny
    h3 = L/nz
    
    j1, j2, j3 = np.meshgrid(np.arange(0, nx, 1),
                             np.arange(0, ny, 1),
                             np.arange(0, nz, 1))

    j1 = np.moveaxis(j1, 0, 1)
    j2 = np.moveaxis(j2, 0, 1)
    j3 = np.moveaxis(j3, 0, 1)

    x1 = (j1+S)*h1
    x2 = (j2+S)*h2

g = grid(nx,ny,nz,L,ndims)
xi = g.init_XI()

plt.title(f"Fibre radius: avg={r_avg*1e6:.2f}μm, min={r_min*1e6:.2f}μm, max={r_max*1e6:.2f}μm, SD={sigma*1e6:.2f}μm")
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 , x1, f_position)
plt.show()

E_n = Lippmann_Schwinger_Fibre_Gauss(C0, N, xi, c_f, c_m, f_position, E_bar, epsilon, ndims, kdiff_fibre)

y_index = int(25e-6/h2)
x_coords = np.linspace(0, L, N+1)

plt.title(f'FFT Gradient Temperature Profile Along y = {25e-6*1e6:.2f} μm')
plt.xlabel('x-coordinate')
plt.ylabel(r'$\nabla$ T')
plt.plot(x_coords*1e6, E_n[y_index, :, 0], label='$E_1(x)$')
plt.plot(x_coords*1e6, E_n[y_index, :, 1], label='$E_2(x)$')
plt.legend()
plt.grid(True)
plt.show()

plt.title('FFT $E_1(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, E_n[...,0])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")
plt.axhline(y=25e-6*1e6 , color='white', linestyle='--', label=f'y = {25e-6*1e6:.2f} μm')
plt.legend(loc=3)
plt.tight_layout()
plt.show()

plt.title('FFT $E_2(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, E_n[...,1])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")
plt.axhline(y=25e-6*1e6, color='white', linestyle='--', label=f'y = {25e-6*1e6:.2f} μm')
plt.legend(loc=3)
plt.tight_layout()
plt.show()

end = time.time()
runtime = end-start
print(runtime)
