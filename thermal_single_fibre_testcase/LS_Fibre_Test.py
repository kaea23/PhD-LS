# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:05:08 2025

@author: kaea23
"""

import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt

def Lippmann_Schwinger_Fibre_Gauss(C0, N, xi, c_f, c_m, f_position, E_bar, epsilon, ndims, kdiff_fibre ):
    # E_0 = (N,2) = (nx,ny,2)
    # E_0 = E_bar
    N = N+1
    E_0 = np.zeros((N,N,ndims)) 

    E_0[...,0] = E_bar[0]
    E_0[...,1] = E_bar[1]
    
    E_n =  np.zeros_like(E_0)
    
    E_tilde =  np.zeros((N,N,ndims)) 
    E_tilde_hat = np.zeros_like(E_0, dtype=np.complex128)

    tau = np.zeros((N,N,ndims))
    tau_hat = np.zeros_like(tau, dtype=np.complex128)
    
    G = Green_operator(xi, N, ndims)
    G = G/C0
    G[0,0,...] = 0
    
    c = np.where(f_position == np.log(kdiff_fibre),c_f , c_m)
    
    # τ0 (x) ← c(x) · E0(x) − C0 * E0(x)
    for i in range(ndims):
        tau[...,i] = c * E_0[...,i] - C0 * E_0[...,i]

    iteration = 0
    
    while True:
        iteration += 1
        
        if np.isnan(tau).any() or np.isinf(tau).any():
            print("NaN or inf detected in tau at iteration %d" %(iteration))
            break
        
        for i in range(ndims):
            tau_hat[...,i] = fftn(tau[...,i], workers=-1)
            
        # Apply Green operator
        if ndims == 2:
            E_tilde_hat[...,0] = - ( G[...,0] * tau_hat[...,0] + G[...,2] * tau_hat[...,1] )
            E_tilde_hat[...,1] = - ( G[...,2] * tau_hat[...,0] + G[...,1] * tau_hat[...,1] )
            
        if ndims == 3:
            E_tilde_hat[...,0] = - ( G[...,0] * tau_hat[...,0] + G[...,5] * tau_hat[...,1] + G[...,4] * tau_hat[...,2] )
            E_tilde_hat[...,1] = - ( G[...,5] * tau_hat[...,0] + G[...,1] * tau_hat[...,1] + G[...,3] * tau_hat[...,2] )
            E_tilde_hat[...,2] = - ( G[...,4] * tau_hat[...,0] + G[...,3] * tau_hat[...,1] + G[...,2] * tau_hat[...,2] )
            
        #IFFT
        for i in range(ndims):
            E_tilde[...,i] = ifftn(E_tilde_hat[...,i]).real
        
        for i in range(ndims):
            E_n[...,i] = E_bar[i] + E_tilde[...,i]

        if np.isnan(E_n).any() or np.isinf(E_n).any():
            print("NaN or inf detected in E_n at iteration %d" %(iteration))
            break

        #crit = ||E_n - E_0||
        crit = np.sqrt(np.sum((E_n - E_0)**2))

        if crit < epsilon:
            print('  Iteration %d - crit (%.3e) smaller than epsilon '%(iteration, crit))
            break
        
        # τn (x) ← c(x) · En(x) − C0 * En(x)
        for i in range(ndims):
            tau[...,i] = c * E_n[...,i] - C0 * E_n[...,i]

        E_0 = np.copy(E_n)
                
    return E_n


def Lippmann_Schwinger_Fibre(C0, N, xi, c_f, c_m, f_position, E_bar, epsilon,
                             L, ndims):
    # E_0 = (N,2) = (nx,ny,2)
    # E_0 = E_bar
    E_0 = np.zeros((N,N,ndims)) 
    E_0[...,0] = E_bar[0]
    E_0[...,1] = E_bar[1]
    
    E_n =  np.zeros_like(E_0)
    
    E_tilde =  np.zeros((N,N,ndims)) 
    E_tilde_hat = np.zeros_like(E_0, dtype=np.complex128)

    tau = np.zeros((N,N,ndims))
    tau_hat = np.zeros_like(tau, dtype=np.complex128)
    
    G = Green_operator(xi, N, ndims)
    G = G/C0
    G[0,0,...] = 0
    
    C = np.where(f_position,c_f,c_m)
    
    # τ0 (x) ← c(x) · E0(x) − C0 * E0(x)
    for i in range(ndims):
        tau[...,i] = C * E_0[...,i] - C0 * E_0[...,i]

    iteration = 0
    
    while True:
        iteration += 1
        
        if np.isnan(tau).any() or np.isinf(tau).any():
            print("NaN or inf detected in tau at iteration %d" %(iteration))
            break
        
        for i in range(ndims):
            tau_hat[...,i] = fftn(tau[...,i], workers=-1)   
        
        # Apply Green operator
        if ndims == 2:
            E_tilde_hat[...,0] = - ( G[...,0] * tau_hat[...,0] + G[...,2] * tau_hat[...,1] )
            E_tilde_hat[...,1] = - ( G[...,2] * tau_hat[...,0] + G[...,1] * tau_hat[...,1] )
            
        if ndims == 3:
            E_tilde_hat[...,0] = - ( G[...,0] * tau_hat[...,0] + G[...,5] * tau_hat[...,1] + G[...,4] * tau_hat[...,2] )
            E_tilde_hat[...,1] = - ( G[...,5] * tau_hat[...,0] + G[...,1] * tau_hat[...,1] + G[...,3] * tau_hat[...,2] )
            E_tilde_hat[...,2] = - ( G[...,4] * tau_hat[...,0] + G[...,3] * tau_hat[...,1] + G[...,2] * tau_hat[...,2] )
            
        #IFFT
        for i in range(ndims):
            E_tilde[...,i] = ifftn(E_tilde_hat[...,i]).real
        
        for i in range(ndims):
            E_n[...,i] = E_bar[i] + E_tilde[...,i] 
        
        if np.isnan(E_n).any() or np.isinf(E_n).any():
            print("NaN or inf detected in E_n at iteration %d" %(iteration))
            break

        #crit = ||E_n - E_0||
        crit = np.sqrt(np.sum((E_n - E_0)**2))

        if crit < epsilon:
            print('  Iteration %d - crit (%.3e) smaller than epsilon '%(iteration, crit))
            break
        
        # τn (x) ← c(x) · En(x) − C0 * En(x)
        for i in range(ndims):
            tau[...,i] = C * E_n[...,i] - C0 * E_n[...,i]

        E_0 = np.copy(E_n)
                
    return E_n, C


def Green_operator(xi, N, ndims):
    
    vdims = int((ndims**2 + ndims) / 2)
    xi_xi = np.zeros((N,N,vdims))
    
    if ndims == 2:
        # (ξ·ξ)/||(ξ)||^2
        xi_xi[...,0] =  xi[...,0]*xi[...,0]
        xi_xi[...,1] =  xi[...,1]*xi[...,1]
        xi_xi[...,2] =  xi[...,0]*xi[...,1]
        
        #green operator
        norm2 = xi[...,0:1]**2 + xi[...,1:2]**2
        
            
    if ndims == 3:
        # (ξ·ξ)/||(ξ)||^2
        xi_xi[...,0] =  xi[...,0]*xi[...,0]
        xi_xi[...,1] =  xi[...,1]*xi[...,1]
        xi_xi[...,2] =  xi[...,2]*xi[...,2]
        xi_xi[...,3] =  xi[...,1]*xi[...,2]
        xi_xi[...,4] =  xi[...,0]*xi[...,2]
        xi_xi[...,5] =  xi[...,0]*xi[...,1]
        
        #green operator
        norm2 = xi[...,0:1]**2 + xi[...,1:2]**2 + xi[...,2:3]**2
        
    norm2[0,0] = 1.
    G = xi_xi / norm2 #keep axis
        
    return G

def Result_test(scheme, C, N, E_n, L, E_bar):
    
    if scheme == 'staggered':
        #DIVERGENCE
        # Step 1: Define conductivity field C(x)
        #C = np.where(f_position == 1, c_f, c_m)

        nx, ny = N
        h = L/N
        D = np.zeros((nx - 1, ny - 1))  # divergence at cell centers 
        
        term1 = C[1:, :-1] * E_n[1:, :-1, 0] - C[:-1, :-1] * E_n[:-1, :-1, 0]
        term2 = C[:-1, 1:] * E_n[:-1, 1:, 1] - C[:-1, :-1] * E_n[:-1, :-1, 1]
        D = (term1 + term2) / h
        
        mean_D = np.mean(D)
        max_D = np.max(D)
        min_D = np.min(D)
        std_D = np.std(D)
        
        #ROTATION        
        R = (1/h) * (
            E_n[1:, 1:, 0] -     # E1[i, j]
            E_n[1:, 1:, 1] -     # E2[i, j]
            E_n[1:, :-1, 0] +    # E1[i, j-1]
            E_n[:-1, 1:, 1]      # E2[i-1, j]
        )
        
        mean_R = np.mean(R)
        max_R = np.max(R)
        min_R = np.min(R)
        std_R = np.std(R)
        
        mean_E1 = np.mean(E_n[...,0])
        mean_E2 = np.mean(E_n[...,1])
        
        print(f"{mean_E1:10.4e}")
        print(f"{mean_E2:10.4e}")

        print("Summary Statistics Comparison: Staggered Scheme")
        print("-" * 40)
        print(f"{'Field':<12} {'Mean':>10} {'Max':>10} {'Min':>10} {'Std Dev':>12}")
        print("-" * 40)
        print(f"{'Rotation':<12} {mean_R:10.4e} {max_R:10.4e} {min_R:10.4e} {std_R:12.4e}")
        print(f"{'Divergence':<12} {mean_D:10.4e} {max_D:10.4e} {min_D:10.4e} {std_D:12.4e}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot the divergence field
        im1 = axes[0].imshow(D.T, origin='lower', cmap='seismic', extent=[0, L, 0, L])
        fig.colorbar(im1, ax=axes[0], label='Divergence')
        axes[0].set_title('Divergence Field D (Staggered Scheme)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # Plot the rotation field
        im2 = axes[1].imshow(R.T, origin='lower', cmap='seismic', extent=[0, L, 0, L])
        fig.colorbar(im2, ax=axes[1], label='Rotation')
        axes[1].set_title('Rotation Field R (Staggered Scheme)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')

        plt.tight_layout()
        plt.show()
        
    elif scheme == 'h':
        # Grid size and spacing 
        nx, ny = N
        h = L/N

        # Conductivity field
        #C = np.where(f_position == 1, c_f, c_m)

        # Initialize divergence field
        D = np.zeros((nx - 1, ny - 1))
        
        term1 = C[1:, :-1] * E_n[1:, :-1, 0] + C[1:, 1:] * E_n[1:, 1:, 0]
        term2 = C[:-1, :-1] * E_n[:-1, :-1, 0] + C[:-1, 1:] * E_n[:-1, 1:, 0]
        term3 = C[:-1, 1:] * E_n[:-1, 1:, 1] + C[1:, 1:] * E_n[1:, 1:, 1]
        term4 = C[:-1, :-1] * E_n[:-1, :-1, 1] + C[1:, :-1] * E_n[1:, :-1, 1]

        D = (term1 - term2 + term3 - term4) / (2 * h)


        # Compute divergence at mesh vertices
        for j1 in range(1, nx): 
            for j2 in range(1, ny): 
                term1 = C[j1, j2 - 1] * E_n[j1, j2 - 1, 0] + C[j1, j2] * E_n[j1, j2, 0]
                term2 = C[j1 - 1, j2 - 1] * E_n[j1 - 1, j2 - 1, 0] + C[j1 - 1, j2] * E_n[j1 - 1, j2, 0]
                term3 = C[j1 - 1, j2] * E_n[j1 - 1, j2, 1] + C[j1, j2] * E_n[j1, j2, 1]
                term4 = C[j1 - 1, j2 - 1] * E_n[j1 - 1, j2 - 1, 1] + C[j1, j2 - 1] * E_n[j1, j2 - 1, 1]

                D[j1 - 1, j2 - 1] = (term1 - term2 + term3 - term4) / (2 * h) 
                
        mean_D = np.mean(D)
        max_D = np.max(D)
        min_D = np.min(D)
        std_D = np.std(D)
        
        #ROTATION
        R = (1 / (2 * h)) * (
            E_n[1:, 1:, 0] +    # E1[i, j]
            E_n[:-1, 1:, 0] -   # E1[i-1, j]
            E_n[1:, :-1, 0] -   # E1[i, j-1]
            E_n[:-1, :-1, 0] -  # E1[i-1, j-1]
            E_n[1:, :-1, 1] -   # E2[i, j-1]
            E_n[1:, 1:, 1] +    # E2[i, j]
            E_n[:-1, :-1, 1] +  # E2[i-1, j-1]
            E_n[:-1, 1:, 1]     # E2[i-1, j]
            )
        
        mean_R = np.mean(R)
        max_R = np.max(R)
        min_R = np.min(R)
        std_R = np.std(R)
        
        mean_E1 = np.mean(E_n[...,0])
        mean_E2 = np.mean(E_n[...,1])
        
        print(f"{mean_E1:10.4e}")
        print(f"{mean_E2:10.4e}")

        print("Summary Statistics Comparison: H-Scheme")
        print("-" * 40)
        print(f"{'Field':<12} {'Mean':>10} {'Max':>10} {'Min':>10} {'Std Dev':>12}")
        print("-" * 40)
        print(f"{'Rotation':<12} {mean_R:10.4e} {max_R:10.4e} {min_R:10.4e} {std_R:12.4e}")
        print(f"{'Divergence':<12} {mean_D:10.4e} {max_D:10.4e} {min_D:10.4e} {std_D:12.4e}")
 
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot the divergence field
        im1 = axes[0].imshow(D.T, origin='lower', cmap='seismic', extent=[0, L, 0, L])
        fig.colorbar(im1, ax=axes[0], label='Divergence')
        axes[0].set_title('Divergence Field D (Staggered Scheme)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # Plot the rotation field
        im2 = axes[1].imshow(R.T, origin='lower', cmap='seismic', extent=[0, L, 0, L])
        fig.colorbar(im2, ax=axes[1], label='Rotation')
        axes[1].set_title('Rotation Field R (Staggered Scheme)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')

        plt.tight_layout()
        plt.show()
    
    return 
     
