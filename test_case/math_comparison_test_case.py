# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:17:35 2025

@author: kaea23
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Parameters
R = 0.25       # Radius of the fibre
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
Ex = np.where(RHO < 0.001, np.nan, Ex)
Ey = np.where(RHO < 0.001, np.nan, Ey)

#Plot Ex component
plt.figure(figsize=(8, 6))
contour_ex = plt.contourf(X, Y, Ex, levels=500, cmap='seismic', vmin=1.4, vmax=2.8)
cbar = plt.colorbar(contour_ex)
cbar.set_label(r"$\nabla V_{E}$ ($ V $)", fontsize=14)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
circle_ex = plt.Circle((0, 0), R, color='black', fill=False, linewidth=0.2)
plt.gca().add_patch(circle_ex)
plt.title('Electric Field Ex Component', fontsize=14)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

# Plot Ey component
plt.figure(figsize=(8, 6))
contour_ey = plt.contourf(X, Y, Ey, levels=500, cmap='seismic', vmin=-0.7, vmax=0.7)
cbar = plt.colorbar(contour_ey)
cbar.set_label(r"$\nabla V_{E}$ ($ V $)", fontsize=14)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
circle_ey = plt.Circle((0, 0), R, color='black', fill=False, linewidth=0.2)
plt.gca().add_patch(circle_ey)
plt.title('Electric Field Ey Component', fontsize=14)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
