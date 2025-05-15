import numpy as np
from scipy.fft import fftn, ifftn
#import sys

def Lippmann_Schwinger(C0, N, xi, c, E_bar, L, epsilon):
    ndims = 2
    # E_0 = (N,2) = (nx,ny,2)
    #E_0 = E_bar
    E_0 = np.zeros((N,N,ndims)) 
    E_0[...,0] = E_bar[0]
    E_0[...,1] = E_bar[1]
    
    E_n =  np.zeros_like(E_0)
    E_tilde =  np.zeros_like(E_0)
    E_tilde_hat = np.zeros_like(E_0, dtype=np.complex128)
    
    tau = np.zeros((N,N,ndims))
    #tau_n = np.zeros_like(tau)
    tau_hat = np.zeros_like(tau, dtype=np.complex128)
    
    # (ξ·ξ)/||(ξ)||^2
    xi_xi = np.zeros((N,N,3))
    xi_xi[...,0] =  xi[...,0]*xi[...,0]
    xi_xi[...,2] =  xi[...,0]*xi[...,1]
    xi_xi[...,1] =  xi[...,1]*xi[...,1]
    
    #green operator
    norm2 = xi[...,0:1]**2 + xi[...,1:2]**2
    norm2[0,0] = 1.
    G = xi_xi / norm2 #keep axis
    G = G/C0
    
    # τ0 (x) ← c(x) · E0(x) − C0 * E0(x) 
    #tau = C_dot_E(c, E_0) - C0 * E_0
    for i in range(ndims):
        tau[...,i] = c * E_0[...,i] - C0* E_0[...,i]
    
    iteration = 0
    
    while True:
        iteration += 1 
        
        for i in range(ndims):
            tau_hat[...,i] = fftn(tau[...,i], workers=-1)
        
        #print("tau_hat:", tau_hat.shape)
        #print(tau_hat)
        
        # Apply Green operator
        E_tilde_hat[...,0] = - ( G[...,0] * tau_hat[...,0] + G[...,2] * tau_hat[...,1] )
        E_tilde_hat[...,1] = - ( G[...,2] * tau_hat[...,0] + G[...,1] * tau_hat[...,1] )
        
        #print("E_tilde_hat:", E_tilde_hat.shape)
        #print(E_tilde_hat)
        
        #IFFT
        E_tilde[...,0] = ifftn(E_tilde_hat[...,0]).real
        E_tilde[...,1] = ifftn(E_tilde_hat[...,1]).real 
        
        print("E_tilde:", E_tilde.shape, E_tilde.dtype )
        print(E_tilde)
        
        for i in range(ndims):
            E_n[...,i] = E_bar[i] + E_tilde[...,i]
        
        print("E_bar:", E_bar.shape, E_bar.dtype)
        print(E_bar)
        
        print("E_n:", E_n.shape, E_n.dtype)
        print(E_n)
        
        #crit = ||E_n - E_0||
        crit = np.sqrt(np.sum((E_n - E_0)**2))
        
        if iteration > 100:
            print('  Iteration max %d - crit (%.3e)  '%(iteration, crit))
            break
        
        if crit < epsilon:
            print('  Iteration %d - crit (%.3e) smaller than epsilon '%(iteration, crit))
            break
        
        
        # τn (x) ← c(x) · En(x) − C0 * En(x)
        for i in range(ndims):
            tau[...,i] = c * E_n[...,i] - C0* E_n[...,i]
        
        #print("tau_n:", tau_n.shape)
        #print(tau_n)
        
        E_0 = np.copy(E_n)
                
    return E_n


def C_dot_E(c,E): #check if this works correctly
    
    # c matrix will have appearance of c(x)I of dim n x n
    # E will be (n,n,2)
    # c_dot_E should be (n,n,2)
    
    c_E = np.array([c*E[...,0],  
                    c*E[...,1]])
    
    return np.einsum('ijk...->kji...', c_E)
    
    
def C_x(x1,x2,L):
    
    # c matrix will have appearance of c(x)I of dim n x n = nx x ny
    
    k1 = (2. * np.pi * x1 / L)
    k2 = (2. * np.pi * x2 / L)
    c = 1/(0.5 + (np.cos(k1))**2 + (np.cos(k2))**2 + (np.cos(k1))**4 + (np.cos(k2))**6 )
            
    return c

def D_x(c,xi,h):
    
    D = 1/h
    
    return D


"""
def C_dot_E(c,E): #check if this works correctly
    
    # c matrix will have appearance of c(x)I of dim n x n
    # E will be (n,1) 
    # c_dot_E should be (n,1)
    
    E = E[...,np.newaxis]
    c_dot_E = np.zeros_like(E)
    
    for row in range (0,(np.size(c, 0))):
        for col in range (0,(np.size(c, 1))):
            
            if (row == col):
                c_dot_E[row] = c[row,col] * E[row]
    
    return c_dot_E
"""