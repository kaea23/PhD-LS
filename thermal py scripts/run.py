import numpy as np
import matplotlib.pyplot as plt
#import time
#import sys
#sys.path.append('H:/phase_field/phase_field_alt/')
from discretisation import grid
from LS import Lippmann_Schwinger, C_x

#start = time.time()

# geometry
L = 1 #m
nx, ny = 11, 11
N = 11
h1 = L/nx
h2 = L/ny

#print(h1, h1)

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

#print("J1:",j1)
#print("J2:",j2)


S1 = 0.5
S2 = 0.5

x1 = (j1+S1)*h1
x2 = (j2+S2)*h2

#print("x1:",x1)
#print("x2:",x2)

#print(x1)
#print(x1.shape)

#print(x2)
#print(x2.shape)


plt.plot(j1, j2, marker='.', color='k', linestyle='none')
plt.plot(x1, x2, marker='x', color='k', linestyle='none')
plt.show()

c = C_x(x1,x2,L)
#print(c)
#plt.imshow(c)
#plt.plot(j1, j2, marker='.', color='k', linestyle='none')
plt.plot(x1, x2, marker='x', color='k', linestyle='none')
plt.pcolor(x1, x2, c)
plt.show()

E_n = Lippmann_Schwinger(C0, N, xi, c, E_bar, L, epsilon)
plt.pcolor(x1, x2, E_n[...,0])
plt.show()