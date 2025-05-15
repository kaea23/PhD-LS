import numpy as np
import matplotlib.pyplot as plt
#import time
#import sys
#sys.path.append('H:/phase_field/phase_field_alt/')
from discretisation import grid
from LS_v1 import Lippmann_Schwinger, C_x

#start = time.time()

# geometry
L = 1 #m
nx, ny = 128, 128
N = 128
h1 = L/nx
h2 = L/ny

#parameters
C0 = 1.5
E_bar = np.array([2, 1], dtype = np.float64)
epsilon = 1e-4

g = grid(nx,ny,L)
xi = g.init_XI()

j1, j2, = np.meshgrid(np.arange(0, nx, 1),
                      np.arange(0, ny, 1))

j1 = np.moveaxis(j1, 0, 1)
j2 = np.moveaxis(j2, 0, 1)

S1 = 0.5
S2 = 0.5

x1 = (j1+S1)*h1
x2 = (j2+S2)*h2

#plt.plot(j1, j2, marker='.', color='k', linestyle='none')
#plt.plot(x1, x2, marker='x', color='k', linestyle='none')
#plt.show()

c = C_x(x1,x2,L)

#plt.plot(j1, j2, marker='.', color='k', linestyle='none')
#plt.plot(x1, x2, marker='x', color='k', linestyle='none')
#plt.pcolor(x1, x2, c)
#plt.show()


E_n = Lippmann_Schwinger(C0, N, xi, c, E_bar, L, epsilon)

#Testing

Average_E_bar_1 =  (np.sum(E_n[...,0]))/ N**2 
Average_E_bar_2 =  (np.sum(E_n[...,1]))/ N**2 

print("Average_E_bar:", Average_E_bar_1, Average_E_bar_2 )



plt.title('$E_1(x)$')
plt.pcolor(x1, x2, E_n[...,0])
plt.colorbar()
plt.show()

plt.title('$E_2(x)$')
plt.pcolor(x1, x2, E_n[...,1])
plt.colorbar()
plt.show()
