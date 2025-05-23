import numpy as np
from scipy.fft import fftn, ifftn

def Lippmann_Schwinger_Fibre(C0, N, xi, c_f, c_m, f_position1, E_bar, L, epsilon):
    ndims = 2
    # E_0 = (N,2) = (nx,ny,2)
    # E_0 = E_bar
    E_0 = np.zeros((N,N,ndims)) 
    E_0[...,0] = E_bar[0]
    E_0[...,1] = E_bar[1]
    
    E_n =  np.zeros_like(E_0)
    
    #E_tilde_m =  np.zeros_like(E_0)
    E_tilde =  np.zeros_like(E_0)
    #E_tilde_m_hat = np.zeros_like(E_0, dtype=np.complex128)
    E_tilde_hat = np.zeros_like(E_0, dtype=np.complex128)
    
    #tau_m = np.zeros((N,N,ndims))
    tau = np.zeros((N,N,ndims))
    #tau_m_hat = np.zeros_like(tau_m, dtype=np.complex128)
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
    
    if f_position1.any == 1:
        # τ0 (x) ← c(x) · E0(x) − C0 * E0(x)
        for i in range(ndims):
            tau[...,i] = c_f * E_0[...,i] - C0* E_0[...,i]
    else:
        # τ0 (x) ← c(x) · E0(x) − C0 * E0(x)
        for i in range(ndims):
            tau[...,i] = c_m * E_0[...,i] - C0* E_0[...,i]
 
    iteration = 0
    
    while True:
        iteration += 1 
        
        for i in range(ndims):
            tau_hat[...,i] = fftn(tau[...,i], workers=-1)
        
        # Apply Green operator
        E_tilde_hat[...,0] = - ( G[...,0] * tau_hat[...,0] + G[...,2] * tau_hat[...,1] )
        E_tilde_hat[...,1] = - ( G[...,2] * tau_hat[...,0] + G[...,1] * tau_hat[...,1] )
        
        #IFFT
        E_tilde[...,0] = ifftn(E_tilde_hat[...,0]).real
        E_tilde[...,1] = ifftn(E_tilde_hat[...,1]).real
        
        
        for i in range(ndims):
            E_n[...,i] = E_bar[i] + E_tilde[...,i]
        
        #crit = ||E_n - E_0||
        crit = np.sqrt(np.sum((E_n - E_0)**2))

        if crit < epsilon:
            print('  Iteration %d - crit (%.3e) smaller than epsilon '%(iteration, crit))
            break
        
        if f_position1 == 1:
            # τ0 (x) ← c(x) · E0(x) − C0 * E0(x)
            for i in range(ndims):
                tau[...,i] = c_f * E_n[...,i] - C0* E_n[...,i]
        else:
            # τ0 (x) ← c(x) · E0(x) − C0 * E0(x)
            for i in range(ndims):
                tau[...,i] = c_m * E_n[...,i] - C0* E_n[...,i]

        E_0 = np.copy(E_n)
                
    return E_n
       