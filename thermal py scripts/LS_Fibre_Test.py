# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:05:08 2025

@author: kaea23
"""

import numpy as np
from scipy.fft import fftn, ifftn

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
    
    # (ξ·ξ)/||(ξ)||^2
    #xi_xi = np.zeros((N,N,3))
    #xi_xi[...,0] =  xi[...,0]*xi[...,0]
    #xi_xi[...,2] =  xi[...,0]*xi[...,1]
    #xi_xi[...,1] =  xi[...,1]*xi[...,1]
    
    #green operator
    #norm2 = xi[...,0:1]**2 + xi[...,1:2]**2
    #norm2[0,0] = 1.
    #G = xi_xi / norm2 #keep axis
    
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
        #E_tilde_hat[...,0] = - ( G[...,0] * tau_hat[...,0] + G[...,2] * tau_hat[...,1] )
        #E_tilde_hat[...,1] = - ( G[...,2] * tau_hat[...,0] + G[...,1] * tau_hat[...,1] )
        
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
        
        #E_tilde[...,0] = ifftn(E_tilde_hat[...,0]).real
        #E_tilde[...,1] = ifftn(E_tilde_hat[...,1]).real
        
        for i in range(ndims):
            E_n[...,i] = E_bar[i] + E_tilde[...,i]

        if np.isnan(E_n).any() or np.isinf(E_n).any():
            print("NaN or inf detected in E_n at iteration %d" %(iteration))
            break

        #crit = ||E_n - E_0||
        crit = np.sqrt(np.sum((E_n - E_0)**2))
        #print('crit', crit) 

        if crit < epsilon:
            print('  Iteration %d - crit (%.3e) smaller than epsilon '%(iteration, crit))
            break
        
        # τn (x) ← c(x) · En(x) − C0 * En(x)
        for i in range(ndims):
            tau[...,i] = c * E_n[...,i] - C0 * E_n[...,i]

        E_0 = np.copy(E_n)
                
    return E_n


def Lippmann_Schwinger_Fibre(C0, N, xi, c_f, c_m, f_position, E_bar, epsilon, ndims):
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
    
    # (ξ·ξ)/||(ξ)||^2
    #xi_xi = np.zeros((N,N,3))
    #xi_xi[...,0] =  xi[...,0]*xi[...,0]
    #xi_xi[...,2] =  xi[...,0]*xi[...,1]
    #xi_xi[...,1] =  xi[...,1]*xi[...,1]
    
    #green operator
    #norm2 = xi[...,0:1]**2 + xi[...,1:2]**2
    #norm2[0,0] = 1.
    #G = xi_xi / norm2 #keep axis
    
    G = Green_operator(xi, N, ndims)
    G = G/C0
    G[0,0,...] = 0
    
    c = np.where(f_position,c_f,c_m)
    
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
        #E_tilde_hat[...,0] = - ( G[...,0] * tau_hat[...,0] + G[...,2] * tau_hat[...,1] )
        #E_tilde_hat[...,1] = - ( G[...,2] * tau_hat[...,0] + G[...,1] * tau_hat[...,1] )
        
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
        
        #E_tilde[...,0] = ifftn(E_tilde_hat[...,0]).real
        #E_tilde[...,1] = ifftn(E_tilde_hat[...,1]).real
        
        for i in range(ndims):
            E_n[...,i] = E_bar[i] + E_tilde[...,i] 
        
        if np.isnan(E_n).any() or np.isinf(E_n).any():
            print("NaN or inf detected in E_n at iteration %d" %(iteration))
            break

        #crit = ||E_n - E_0||
        crit = np.sqrt(np.sum((E_n - E_0)**2))
        #print('crit', crit) 

        if crit < epsilon:
            print('  Iteration %d - crit (%.3e) smaller than epsilon '%(iteration, crit))
            break
        
        # τn (x) ← c(x) · En(x) − C0 * En(x)
        for i in range(ndims):
            tau[...,i] = c * E_n[...,i] - C0 * E_n[...,i]

        E_0 = np.copy(E_n)
                
    return E_n


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
     
