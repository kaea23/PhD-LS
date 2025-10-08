# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:17:35 2025

@author: kaea23
"""


import numpy as np
import matplotlib.pyplot as plt


# Parameters
R = 0.25       # Radius of the fibre
E0x = 2.0      # External field magnitude in x-direction
E0y = 0.0      # External field magnitude in y-direction
kappa = 0.5    # Conductivity ratio
# Define Cartesian grid
x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(x, y, indexing='ij')


# Convert to polar coordinates
RHO = np.sqrt(X**2 + Y**2)
THETA = np.arctan2(Y, X)

# Unit vectors in polar coordinates
cos_theta = np.cos(THETA)
sin_theta = np.sin(THETA)


# Electric field components in polar coordinates
Ein_rho = (2 * kappa / (1 + kappa)) * (E0x * cos_theta + E0y * sin_theta)
Ein_phi = -(2 * kappa / (1 + kappa)) * (-E0x * sin_theta + E0y * cos_theta)


Eout_rho = (1 + (1 - kappa) / (1 + kappa) * (R**2 / RHO**2)) * (E0x * cos_theta + E0y * sin_theta)
Eout_phi = -(1 - (1 - kappa) / (1 + kappa) * (R**2 / RHO**2)) * (-E0x * sin_theta + E0y * cos_theta)

# Combine based on region
Erho = np.where(RHO <= R, Ein_rho, Eout_rho)
Ephi = np.where(RHO <= R, Ein_phi, Eout_phi)

# Convert to Cartesian components
Ex = Erho * cos_theta - Ephi * sin_theta
Ey = Erho * sin_theta + Ephi * cos_theta

# Mask singularity at origin
Ex = np.where(RHO < 0.001, np.nan, Ex)
Ey = np.where(RHO < 0.001, np.nan, Ey)



#Plot Ex component
plt.figure(figsize=(8, 6))
contour_ex = plt.contourf(X, Y, Ex, levels=100, cmap='seismic')
plt.colorbar(contour_ex, label='Ex Component')
circle_ex = plt.Circle((0, 0), R, color='black', fill=False, linewidth=2)
plt.gca().add_patch(circle_ex)
plt.title('Electric Field Ex Component')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()

# Plot Ey component
plt.figure(figsize=(8, 6))
contour_ey = plt.contourf(X, Y, Ey, levels=100, cmap='seismic')
plt.colorbar(contour_ey, label='Ey Component')
circle_ey = plt.Circle((0, 0), R, color='black', fill=False, linewidth=2)
plt.gca().add_patch(circle_ey)
plt.title('Electric Field Ey Component')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()

##


def mathcomp(E_bar):
    # Parameters
    R = 0.25       # Fibre radius
    E0x = E_bar[0]     # External field in x-direction
    E0y = E_bar[1]     # External field in y-direction
    kappa = 0.5    # Conductivity ratio

    # Define horizontal line y = 0
    x = np.linspace(-1, 1, 1000)
    y = np.zeros_like(x)

    # Convert to polar coordinates
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Unit vectors in polar coordinates
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Electric field components in polar coordinates
    Ein_rho = (2 * kappa / (1 + kappa)) * (E0x * cos_theta + E0y * sin_theta)
    Ein_phi = -(2 * kappa / (1 + kappa)) * (-E0x * sin_theta + E0y * cos_theta)

    Eout_rho = (1 + (1 - kappa) / (1 + kappa) * (R**2 / rho**2)) * (E0x * cos_theta + E0y * sin_theta)
    Eout_phi = -(1 - (1 - kappa) / (1 + kappa) * (R**2 / rho**2)) * (-E0x * sin_theta + E0y * cos_theta)

    # Combine based on region
    Erho = np.where(rho <= R, Ein_rho, Eout_rho)
    Ephi = np.where(rho <= R, Ein_phi, Eout_phi)

    # Convert to Cartesian components
    Ex = Erho * cos_theta - Ephi * sin_theta
    #Ey = Erho * sin_theta + Ephi * cos_theta
    
    #correction_term = ((1 - kappa) / (1 + kappa)) * (R**2 / x**2)
    
    
    return Ex, R


"""
# Styled plot
plt.figure(figsize=(6, 4))
plt.plot(x, Ex, label='$E_1(x)$', color='blue')
plt.axvline(-R, color='black', linestyle='--', linewidth=1)
plt.axvline(R, color='black', linestyle='--', linewidth=1)

# Title and labels
plt.title('FFT Gradient Temperature Profile Along y = 0')
plt.xlabel('x-coordinate')
plt.ylabel('$\\nabla T$')

# Grid and legend
plt.grid(True)
#plt.legend()

# Show plot
plt.tight_layout()
plt.show()
"""


# Parameters
R = 0.25       # Fibre radius
E_bar = 2.0    # FFT-derived average field
E0x = 2.0      # External field in x-direction (exact solution)
E0y = 0.0      # External field in y-direction
kappa = 0.5    # Conductivity ratio

# Define horizontal line y = 0 excluding fibre region
x = np.linspace(-1, 1, 1000)
mask = (x < -R) | (x > R)
x_masked = x[mask]
y_masked = np.zeros_like(x_masked)

# Convert to polar coordinates
rho = np.sqrt(x_masked**2 + y_masked**2)
theta = np.arctan2(y_masked, x_masked)

# Unit vectors in polar coordinates
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# Electric field components in polar coordinates (exact solution)
Eout_rho = (1 + (1 - kappa) / (1 + kappa) * (R**2 / rho**2)) * (E0x * cos_theta + E0y * sin_theta)
Eout_phi = -(1 - (1 - kappa) / (1 + kappa) * (R**2 / rho**2)) * (-E0x * sin_theta + E0y * cos_theta)

# Convert to Cartesian components
Ex_exact = Eout_rho * cos_theta - Eout_phi * sin_theta

# Compute error: FFT-derived average field minus exact solution
error = E_bar - Ex_exact


# Plot error
plt.figure(figsize=(8, 5))
plt.plot(x_masked, error, color='purple', label='Error: $\\bar{E}_x - E_{0x}$')
plt.axvline(-R, color='black', linestyle='--', linewidth=1)
plt.axvline(R, color='black', linestyle='--', linewidth=1)
plt.title('Error Between FFT-Derived Field and Exact Solution Along y = 0')
plt.xlabel('x-coordinate')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#####

# Parameters
R = 0.25       # Fibre radius
E0 = 2.0       # External field magnitude
kappa = 0.5    # Conductivity ratio

# Define x values excluding the fibre region
x = np.linspace(-1, 1, 1000)
mask = (x < -R) | (x > R)
x_masked = x[mask]

# Compute rho and theta for y = 0
y_masked = np.zeros_like(x_masked)
rho = np.sqrt(x_masked**2 + y_masked**2)
theta = np.arctan2(y_masked, x_masked)

# Unit vectors in polar coordinates
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# Exact analytical solution for Ex outside the fibre
Eout_rho = (1 + (1 - kappa) / (1 + kappa) * (R**2 / rho**2)) * E0 * cos_theta
Eout_phi = -(1 - (1 - kappa) / (1 + kappa) * (R**2 / rho**2)) * (-E0 * sin_theta)
Ex_exact = Eout_rho * cos_theta - Eout_phi * sin_theta

# Expected average field E_bar(x)
E_bar = E0 + ((1 - kappa) / (1 + kappa)) * (R**2 / x_masked**2)

# Compute error
error = E_bar - Ex_exact

# Correction term
correction_term = 2*((1 - kappa) / (1 + kappa)) * (R**2 / x_masked**2)

# Plot error and correction term
plt.figure(figsize=(10, 6))
#plt.plot(x_masked, error, color='purple', label='Error: $\\bar{E}_x(x) - E_x(x)$')
plt.plot(x_masked, correction_term, color='green', linestyle='--',
         label='Correction Term: $\\frac{1 - \\kappa}{1 + \\kappa} \\cdot \\frac{R^2}{\\rho^2}$')
plt.axvline(-R, color='black', linestyle='--', linewidth=1)
plt.axvline(R, color='black', linestyle='--', linewidth=1)
plt.title('Error Term Comparison Along y = 0')
plt.xlabel('x-coordinate')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




####

# Parameters
R = 0.25       # Fibre radius
E0 = 2.0       # External field magnitude
kappa = 0.5    # Conductivity ratio

# Define x values excluding the fibre region
x = np.linspace(-1, 1, 1000)
mask = (x < -R) | (x > R)
x_masked = x[mask]

# Compute rho and theta for y = 0
y_masked = np.zeros_like(x_masked)
rho = np.sqrt(x_masked**2 + y_masked**2)
theta = np.arctan2(y_masked, x_masked)

# Unit vectors in polar coordinates
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# Exact analytical solution for Ex outside the fibre
Eout_rho = (1 + (1 - kappa) / (1 + kappa) * (R**2 / rho**2)) * E0 * cos_theta
Eout_phi = -(1 - (1 - kappa) / (1 + kappa) * (R**2 / rho**2)) * (-E0 * sin_theta)
Ex_exact = Eout_rho * cos_theta - Eout_phi * sin_theta

# Expected average field E_bar(x)
E_bar = E0 + ((1 - kappa) / (1 + kappa)) * (R**2 / x_masked**2)

# Compute error
error = E_bar - Ex_exact

# Plot exact and expected average field
plt.figure(figsize=(10, 6))
plt.plot(x_masked, Ex_exact, label='Exact $E_0(x)$', color='blue')
plt.plot(x_masked, E_bar, label='Expected $\\bar{E}_x(x)$', color='orange')
plt.axvline(-R, color='black', linestyle='--', linewidth=1)
plt.axvline(R, color='black', linestyle='--', linewidth=1)
plt.title('Comparison of Exact and Expected Average Field Along y = 0')
plt.xlabel('x-coordinate')
plt.ylabel('Electric Field')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot error
plt.figure(figsize=(10, 6))
plt.plot(x_masked, error, color='purple', label='Error: $\\bar{E}_x(x) - E_0(x)$')
plt.axvline(-R, color='black', linestyle='--', linewidth=1)
plt.axvline(R, color='black', linestyle='--', linewidth=1)
plt.title('Error Between Expected Average Field and Exact Solution Along y = 0')
plt.xlabel('x-coordinate')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



