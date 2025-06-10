import numpy as np

def Fibres(L, N, f_num, f_r):
    h1, h2 = L/N, L/N
    
    A_mat = L*L
    A_f = f_num*np.pi*(f_r)**2 
    f_percent = (A_f/A_mat)*100
    
    fx_center = np.zeros(f_num)
    fy_center = np.zeros(f_num)
    
    #for i in range (f_num):
    #    fx_center[i] = random.randint(0, int(L * 100)) / 100.0
    #    fy_center[i] = random.randint(0, int(L * 100)) / 100.0
    fx_center = np.random.uniform(low=0, high=L, size=f_num)
    fy_center = np.random.uniform(low=0, high=L, size=f_num)
    
    print('x centre',fx_center) 
    print('y centre',fy_center) 
    
    f_position = np.full((N,N), False, dtype=bool)
    
    #for i in range (f_num):
    #    for j in range (N):
    #        for k in range (N):
    #            x = j * h1
    #            y = k * h2
    #            if (x - fx_center[i])**2 + (y - fy_center[i])**2 <= f_r**2:
    #                f_position[j,k] = 1
        
    x, y = np.meshgrid(np.arange(0,N), np.arange(0,N))
    for i in range(f_num):
        idx = (x*h1-fx_center[i])**2 + (y*h2-fy_center[i])**2 <= f_r**2
        f_position[idx] = 1
    
    print('  Fibre volume percentage %.2f  %%'%(f_percent))
    return f_position, fy_center
    


"""
def Fibre_C_x(x1,x2,f_r):
    
    # c matrix will have appearance of c(x)I of dim n x n = nx x ny
    
    k1 = (2. * np.pi * x1 / f_r)
    k2 = (2. * np.pi * x2 / f_r)
    c = 1/(0.5 + (np.cos(k1))**2 + (np.cos(k2))**2 + (np.cos(k1))**4 + (np.cos(k2))**6 )
            
    return c

def Mat_C_x(x1,x2,L):
    
    # c matrix will have appearance of c(x)I of dim n x n = nx x ny
    
    k1 = (2. * np.pi * x1 / L)
    k2 = (2. * np.pi * x2 / L)
    c = 1/(0.5 + (np.cos(k1))**2 + (np.cos(k2))**2 + (np.cos(k1))**4 + (np.cos(k2))**6 )
            
    return c
"""
