import numpy as np
import matplotlib.pyplot as plt
import time
from discretisation import grid
from distributions_fibres import __all__
from LS_Fibre_Test import Lippmann_Schwinger_Fibre

start = time.time()

# Material Geometry
L = 50e-6 #m 50micrometers
nx, ny = 500, 500
N = 500
h1 = L/nx
h2 = L/ny

# Fibre Geometry
#f_num = 6 #number of fibres
#f_r = 5e-6 #fibre radius m
r_avg = 153e-7 #15.3 micro m
r_min = 94e-7 #9.4 micro m
r_max = 195e-7 #19.5 micro m
sigma = 32e-7 #3.2 micro m SD
gaussian = True

# Parameters
E_bar = np.array([10, 10], dtype = np.float64) #gradT
epsilon = 1e-8
c_f = 22 #W/mK
c_m = 120 #W/mK
C0 = c_f + c_m/2 # too low diverge

g = grid(nx,ny,L)
xi = g.init_XI()

j1, j2 = np.meshgrid(np.arange(0, nx, 1),
                     np.arange(0, ny, 1))

j1 = np.moveaxis(j1, 0, 1)
j2 = np.moveaxis(j2, 0, 1)

S1 = 0.5
S2 = 0.5

x1 = (j1+S1)*h1
x2 = (j2+S2)*h2

"""
f_position, fy_center = Fibres(L, N, f_num, f_r)

plt.title(u"Fibre number 1 radius 5\u03bcm")
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x1, x2, f_position)
plt.show()

E_n = Lippmann_Schwinger_Fibre(C0, N, xi, c_f, c_m, f_position, E_bar, L, epsilon)
#E_n = Lippmann_Schwinger(C0, N, xi, c, E_bar, L, epsilon)

#Testing

#Average_E_bar_1 =  (np.sum(E_n[...,0]))/ N**2 
#Average_E_bar_2 =  (np.sum(E_n[...,1]))/ N**2 

#print("Average_E_bar:", Average_E_bar_1, Average_E_bar_2 )

#y_index = int(fy_center[0]/h2)
y_index = int(25e-6/h2)
x_coords = np.linspace(0, L, N)
plt.plot(x_coords, E_n[:, y_index, 0], label='$E_1(x)$')
plt.plot(x_coords, E_n[:, y_index, 1], label='$E_2(x)$')
plt.xlabel('x-coordinate')
plt.ylabel(r'$\nabla$ T')
#plt.title(f'Temperature Profile along y = {fy_center[0]}')
plt.title('Gradient Temperature Profile Along y = 25e-6')
plt.legend()
plt.grid(True)
plt.show()

plt.title('$E_1(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x1, x2, E_n[...,0])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")
plt.pcolor(x1, x2, E_n[...,0])
#cbar.set_label("K ($W m^-1 K^-1$)")
plt.show()

plt.title('$E_2(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x1, x2, E_n[...,1])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")
plt.show()



y_index = int(fy_center[0]/h2)
x_coords = np.linspace(0, L, N)
plt.plot(x_coords, E_n[y_index, :, 0], label='$E_1(x)$')
plt.plot(x_coords, E_n[y_index, :, 1], label='$E_2(x)$')
plt.xlabel('x-coordinate')
plt.ylabel('Temperature')
plt.title(f'Temperature Profile along y = {fy_center[0]}')
plt.legend()
plt.grid(True)
plt.show()
"""
end = time.time()
runtime = end-start
print(runtime)