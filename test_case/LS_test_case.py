# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:30:40 2025

@author: kaea23
"""

import numpy as np
from scipy.fft import fftn, ifftn

def Lippmann_Schwinger_Fibre(C0, L, N, xi, c_f, c_m, f_position, E_bar, epsilon, ndims, scheme):
    # E_0 = (N,2) = (nx,ny,2)
    # E_0 = E_bar
    E_0 = np.zeros((N,N,ndims)) 
    E_0[...,0] = E_bar[0]
    E_0[...,1] = E_bar[1]
    
    comp_E_0 = np.zeros((N,N,ndims))
    comp_E_0[...,0] = E_bar[0]
    comp_E_0[...,1] = E_bar[1]
    
    E_n =  np.zeros_like(E_0)
    
    E_tilde =  np.zeros((N,N,ndims)) 
    E_tilde_hat = np.zeros_like(E_0, dtype=np.complex128)

    tau = np.zeros((N,N,ndims))
    tau_hat = np.zeros_like(tau, dtype=np.complex128)
    
    G = Green_operator(xi, N, ndims)
    G = G/C0
    G[0,0,...] = 0
    
    C = np.where(f_position, c_f, c_m)
    
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
            print('Iteration %d - crit (%.3e) smaller than epsilon '%(iteration, crit))
            break
        
        # τn (x) ← c(x) · En(x) − C0 * En(x)  
        tau[..., 0] = (C - C0) * E_n[..., 0] 
        tau[..., 1] = (C - C0) * E_n[..., 1] 
        
        E_0 = np.copy(E_n)
    
  
    return E_n, C, comp_E_0


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

     
