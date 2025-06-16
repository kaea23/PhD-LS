import numpy as np
import matplotlib.pyplot as plt
import time
from discretisation import grid
from fibres import Fibres
from LS_Fibre_Test import Lippmann_Schwinger_Fibre
from LS_Fibre_NNBC import Lippmann_Schwinger_Fibre_NNBC
from LS_Fibre_DDBC import Lippmann_Schwinger_Fibre_DDBC

start = time.time()

# Material Geometry
L = 50e-6 #m 50micrometers
nx, ny = 500, 500
N = 500
h1 = L/nx
h2 = L/ny

# Fibre Geometry
f_num = 1 #number of fibres
f_r = 5e-6 #fibre radius m

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

f_position, fy_center = Fibres(L, N, f_num, f_r)

plt.title(f"Fibre number {f_num} radius {f_r*1e6:.2f} μm")
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, f_position)
plt.show()

E_n = Lippmann_Schwinger_Fibre(C0, N, xi, c_f, c_m, f_position, E_bar, L, epsilon)
E_n_NNBC = Lippmann_Schwinger_Fibre_NNBC(C0, N, xi, c_f, c_m, f_position, E_bar, L, epsilon)
E_n_DDBC = Lippmann_Schwinger_Fibre_DDBC(C0, N, xi, c_f, c_m, f_position, E_bar, L, epsilon)

#Testing

#Average_E_bar_1 =  (np.sum(E_n[...,0]))/ N**2 
#Average_E_bar_2 =  (np.sum(E_n[...,1]))/ N**2 

#print("Average_E_bar:", Average_E_bar_1, Average_E_bar_2 )

x_coords = np.linspace(0, L, N)
y_index = int(fy_center[0]/h2)
#y_index = int(25e-6/h2)

#PLOTTING E_N

plt.plot(x_coords * 1e6, E_n[y_index, :, 0], label='$E_1(x)$')
plt.plot(x_coords * 1e6, E_n[y_index, :, 1], label='$E_2(x)$')
plt.xlabel('x-coordinate')
plt.ylabel(r'$\nabla$ T')
plt.title(f'FFT Gradient Temperature Profile Along y = {fy_center[0]*1e6:.2f} μm')
#plt.title('Gradient Temperature Profile Along y = 25e-6')
plt.legend()
plt.grid(True)
plt.show()

plt.title('FFT $E_1(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, E_n[...,0])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")

plt.axhline(y=fy_center[0]* 1e6, color='white', linestyle='--', label=f'y = {fy_center[0]*1e6:.2f} μm')
plt.legend()
plt.tight_layout()

plt.show()

plt.title('FFT $E_2(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, E_n[...,1])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")

plt.axhline(y=fy_center[0]* 1e6, color='white', linestyle='--', label=f'y = {fy_center[0]*1e6:.2f} μm')
plt.legend()
plt.tight_layout()

plt.show()

#PLOTTING E_N_NNBC

plt.plot(x_coords * 1e6, E_n_NNBC[y_index, :, 0], label='$E_1(x)$')
plt.plot(x_coords * 1e6, E_n_NNBC[y_index, :, 1], label='$E_2(x)$')
plt.xlabel('x-coordinate')
plt.ylabel(r'$\nabla$ T')
plt.title(f'DCT1 Gradient Temperature Profile Along y = {fy_center[0]*1e6:.2f} μm')
#plt.title('Gradient Temperature Profile Along y = 25e-6')
plt.legend()
plt.grid(True)
plt.show()

plt.title('DCT1 $ E_1(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, E_n_NNBC[...,0])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")

plt.axhline(y=fy_center[0]* 1e6, color='white', linestyle='--', label=f'y = {fy_center[0]*1e6:.2f} μm')
plt.legend()
plt.tight_layout()

plt.show()

plt.title('DCT1 $ E_2(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, E_n_NNBC[...,1])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")

plt.axhline(y=fy_center[0]* 1e6, color='white', linestyle='--', label=f'y = {fy_center[0]*1e6:.2f} μm')
plt.legend()
plt.tight_layout()

plt.show()

#PLOTTING E_N_DDBC

plt.plot(x_coords * 1e6, E_n_DDBC[y_index, :, 0], label='$E_1(x)$')
plt.plot(x_coords * 1e6, E_n_DDBC[y_index, :, 1], label='$E_2(x)$')
plt.xlabel('x-coordinate')
plt.ylabel(r'$\nabla$ T')
plt.title(f'DST1 Gradient Temperature Profile Along y = {fy_center[0]*1e6:.2f} μm')
#plt.title('Gradient Temperature Profile Along y = 25e-6')
plt.legend()
plt.grid(True)
plt.show()

plt.title('DST1 $ E_1(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, E_n_DDBC[...,0])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")

plt.axhline(y=fy_center[0]* 1e6, color='white', linestyle='--', label=f'y = {fy_center[0]*1e6:.2f} μm')
plt.legend()
plt.tight_layout()

plt.show()

plt.title('DST1 $ E_2(x)$')
plt.xlabel(u"X (\u03bcm)")
plt.ylabel(u"Y (\u03bcm)")
plt.pcolor(x2 * 1e6, x1 * 1e6, E_n_DDBC[...,1])
cbar = plt.colorbar()
cbar.set_label(r"$\nabla$ T ($ K m^{-1} $)")

plt.axhline(y=fy_center[0]* 1e6, color='white', linestyle='--', label=f'y = {fy_center[0]*1e6:.2f} μm')
plt.legend()
plt.tight_layout()

plt.show()

# PLOT FFT, DCT1, DST1

plt.plot(x_coords * 1e6, E_n[y_index, :, 0], label='FFT')
plt.plot(x_coords * 1e6, E_n_NNBC[y_index, :, 0], label='DCT1')
plt.plot(x_coords * 1e6, E_n_DDBC[y_index, :, 0], label='DST1')
plt.xlabel('x-coordinate')
plt.ylabel(r'$\nabla$ T')
plt.title(f'$E_1(x)$ Gradient Temperature Profile Along y = {fy_center[0]*1e6:.2f} μm')
#plt.title('Gradient Temperature Profile Along y = 25e-6')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(x_coords * 1e6, E_n[y_index, :, 1], label='FFT')
plt.plot(x_coords * 1e6, E_n_NNBC[y_index, :, 1], label='DCT1')
plt.plot(x_coords * 1e6, E_n_DDBC[y_index, :, 1], label='DST1')
plt.xlabel('x-coordinate')
plt.ylabel(r'$\nabla$ T')
plt.title(f'$E_2(x)$ Gradient Temperature Profile Along y = {fy_center[0]*1e6:.2f} μm')
#plt.title('Gradient Temperature Profile Along y = 25e-6')
plt.legend()
plt.grid(True)
plt.show()

end = time.time()
runtime = end-start
print(runtime)

