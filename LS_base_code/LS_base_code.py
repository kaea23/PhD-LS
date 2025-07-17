# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 10:54:20 2025

@author: kaea23
"""

import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt

def Lippmann_Schwinger(C0, x1, x2, L, N, xi, E_bar, epsilon, ndims, scheme):
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
    
    C = C_x(x1,x2,L)
    
    # τ0 (x) ← c(x) · E0(x) − C0 * E0(x)    
    tau[..., 0] = (C - C0) * E_0[..., 0] 
    tau[..., 1] = (C - C0) * E_0[..., 1] 
    
   
    iteration = 0
    
    while True:
        iteration += 1
        
        if np.isnan(tau).any() or np.isinf(tau).any():
            print("NaN or inf detected in tau at iteration %d" %(iteration))
            break
            
        #tau_hat[..., :ndims] = fftn(tau[..., :ndims], workers=-1)
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
        #E_tilde[..., :ndims] = ifftn(E_tilde_hat[..., :ndims]).real
        
        #E_n[..., :ndims] = E_bar[:ndims] + E_tilde[..., :ndims]
        
        for i in range(ndims):
           E_tilde[...,i] = ifftn(E_tilde_hat[...,i]).real
       
        for i in range(ndims): 
            E_n[...,i] = E_bar[i] + E_tilde[...,i]

        #crit = ||E_n - E_0||
        crit = np.sqrt(np.sum((E_n - E_0)**2))
       
        if crit < epsilon:
            print('  Iteration %d - crit (%.3e) smaller than epsilon '%(iteration, crit))
            Result_test(scheme, x1, x2, C, N, E_n, L, E_bar)
            break
        
        # τn (x) ← c(x) · En(x) − C0 * En(x)  
        tau[..., 0] = (C - C0) * E_n[..., 0] 
        tau[..., 1] = (C - C0) * E_n[..., 1] 
        
        E_0 = np.copy(E_n)
                
    return E_n, C

def C_x(x1,x2,L):

    # c matrix will have appearance of c(x)I of dim n x n = nx x ny
    
    k1 = (2. * np.pi * x1 / L)
    k2 = (2. * np.pi * x2 / L)
    c = 1/(0.5 + (np.cos(k1))**2 + (np.cos(k2))**2 + (np.cos(k1))**4 + (np.cos(k2))**6 )
            
    return c


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

def Result_test(scheme, x1, x2, C, N, E_n, L, E_bar):
    
    
    if scheme == 'S':
        
        #DIVERGENCE
        nx, ny = N, N
        h = L/N
        D = np.zeros((nx - 1, ny - 1))  # divergence at cell centers 
        
        C_pos = 'C1'
        C1 = C_div(C_pos,x1,x2,N,L)
        
        C_pos = 'C2'
        C2 = C_div(C_pos,x1,x2,N,L)
        
        term1 = (1/h)*(C1[1:, :-1] * E_n[1:, :-1, 0] - C1[:-1, :-1] * E_n[:-1, :-1, 0])
        term2 = (1/h)*(C2[:-1, 1:] * E_n[:-1, 1:, 1] - C2[:-1, :-1] * E_n[:-1, :-1, 1])
        D = term1 + term2
        
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
        
        #MEAN
        Average_E_bar_1 =  (np.sum(E_n[...,0]))/ N**2 
        Average_E_bar_2 =  (np.sum(E_n[...,1]))/ N**2 

        print("Average_E_bar:", Average_E_bar_1, Average_E_bar_2 )

        print("Summary Statistics Comparison: S-Scheme")
        print("-" * 60)
        print(f"{'Field':<12} {'Mean':>10} {'Max':>10} {'Min':>10} {'Std Dev':>12}")
        print("-" * 60)
        print(f"{'Rotation':<12} {mean_R:10.4e} {max_R:10.4e} {min_R:10.4e} {std_R:12.4e}")
        print(f"{'Divergence':<12} {mean_D:10.4e} {max_D:10.4e} {min_D:10.4e} {std_D:12.4e}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot the divergence field
        im1 = axes[0].imshow(D.T, origin='lower', cmap='seismic', extent=[0, L, 0, L])
        fig.colorbar(im1, ax=axes[0], label='Divergence')
        axes[0].set_title('Divergence Field D (S-Scheme)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # Plot the rotation field
        im2 = axes[1].imshow(R.T, origin='lower', cmap='seismic', extent=[0, L, 0, L])
        fig.colorbar(im2, ax=axes[1], label='Rotation')
        axes[1].set_title('Rotation Field R (S-Scheme)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')

        plt.tight_layout()
        plt.show()
        
    elif scheme == 'H':
        
        #DIVERGENCE
        nx, ny = N, N
        h = L/N
        D = np.zeros((nx - 1, ny - 1)) # divergence at cell centers
        
        term1 = C[1:, :-1] * E_n[1:, :-1, 0] + C[1:, 1:] * E_n[1:, 1:, 0]
        term2 = C[:-1, :-1] * E_n[:-1, :-1, 0] + C[:-1, 1:] * E_n[:-1, 1:, 0]
        term3 = C[:-1, 1:] * E_n[:-1, 1:, 1] + C[1:, 1:] * E_n[1:, 1:, 1]
        term4 = C[:-1, :-1] * E_n[:-1, :-1, 1] + C[1:, :-1] * E_n[1:, :-1, 1]

        D = (term1 - term2 + term3 - term4) / (2 * h) 
             
        mean_D = np.mean(D)
        max_D = np.max(D)
        min_D = np.min(D)
        std_D = np.std(D)
        
        #ROTATION
        R = (1/(2*h)) * (
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
        
        #MEAN
        Average_E_bar_1 =  (np.sum(E_n[...,0]))/ N**2 
        Average_E_bar_2 =  (np.sum(E_n[...,1]))/ N**2 

        print("Average_E_bar:", Average_E_bar_1, Average_E_bar_2 )

        print("Summary Statistics Comparison: H-Scheme")
        print("-" * 60)
        print(f"{'Field':<12} {'Mean':>10} {'Max':>10} {'Min':>10} {'Std Dev':>12}")
        print("-" * 60)
        print(f"{'Rotation':<12} {mean_R:10.4e} {max_R:10.4e} {min_R:10.4e} {std_R:12.4e}")
        print(f"{'Divergence':<12} {mean_D:10.4e} {max_D:10.4e} {min_D:10.4e} {std_D:12.4e}")
 
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot the divergence field
        im1 = axes[0].imshow(D.T, origin='lower', cmap='seismic', extent=[0, L, 0, L])
        fig.colorbar(im1, ax=axes[0], label='Divergence')
        axes[0].set_title('Divergence Field D (H-Scheme)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # Plot the rotation field
        im2 = axes[1].imshow(R.T, origin='lower', cmap='seismic', extent=[0, L, 0, L])
        fig.colorbar(im2, ax=axes[1], label='Rotation')
        axes[1].set_title('Rotation Field R (H-Scheme)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')

        plt.tight_layout()
        plt.show()
    
    return 

def C_div(C_pos,x1,x2,N,L):
    
    # c matrix will have appearance of c(x)I of dim n x n = nx x ny
    
    h = L/N
    
    if C_pos == 'C1':
        x1 = x1-0.5*h
        x2 = x2
        c = C_x(x1,x2,L)
        
    elif C_pos == 'C2':
        x1 = x1
        x2 = x2-0.5*h
        c = C_x(x1,x2,L)
         
    return c

     