import numpy as np
import random

def Fibres(L, N, f_num, f_r, j1, j2, S1, S2, h1, h2):
    
    r_crit = 0.5*L
    
    if f_r > r_crit:
        print('  Fibre radius too large, must be less than %d m '%(r_crit))
        
           
    A_mat = L*L
    A_f = f_num*np.pi*(f_r)**2 
    f_percent = (A_f/A_mat)*100
    
    print(A_mat) #good
    print(A_f)   #good
    
    fx_center = np.zeros(f_num)
    fy_center = np.zeros(f_num)
    
    print(fx_center.shape) #good
    
    for i in range (f_num):
        fx_center[i] = random.randint(0, int(L * 100)) / 100.0
        fy_center[i] = random.randint(0, int(L * 100)) / 100.0
    
    print('x centre',fx_center) #good
    print('y centre',fy_center) #good
    
    f_j1 = np.zeros_like(j1)
    f_j2 = np.zeros_like(j2)
    
    print(f_j1.shape) #good
    print(f_j2.shape) #good
    
    for i in range (f_num):
        print('fx:',fx_center[i])
        print('fy:',fy_center[i])
        for j in range (N):
            for k in range (N):
                x = j * h1
                y = k * h2
                if (x - fx_center[i])**2 + (y - fy_center[i])**2 <= f_r**2:
                    f_j1[j,k] = j
                    f_j2[j,k] = k
                    
    print('f_j1',f_j1) #good
    print('f_j1',f_j1) #good        
               
    f_x1 = (f_j1+S1)*h1
    f_x2 = (f_j2+S2)*h2
    
    print('f_x1',f_x1) #good i think
    print('f_x1',f_x1) #good i think
    
    print('  Fibre volume percentage %.2f  %%'%(f_percent))
    
    return f_j1, f_j2, f_x1, f_x2

def Fibre_C_x(f_x1,f_x2,f_r):
    
    # c matrix will have appearance of c(x)I of dim n x n = nx x ny
    
    k1 = (2. * np.pi * f_x1 / f_r)
    k2 = (2. * np.pi * f_x2 / f_r)
    c = 1/(0.5 + (np.cos(k1))**2 + (np.cos(k2))**2 + (np.cos(k1))**4 + (np.cos(k2))**6 )
            
    return c