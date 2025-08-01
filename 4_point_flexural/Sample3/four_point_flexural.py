# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:23:11 2025

@author: kaea23


File : D:\TJMGROUP\YANGCHEN\20250619_SICSIC_BN15\SAMPLE3\FORCE_DISPL.CSV saved 6/22/2025 at 4:33:48 PM
Sample : Default
Span : 10.0
Width : 5.0
Thickness : 1.0
Start Sampletime : 200ms
Start Speed : 0.1mm/min
Gain : x1
Video file : 
Scale : 4.078
Number of points : 737
Ver6.3.45
Compression

Test duration : 147.8s
Max travel : 2.828mm
Max extension : 0.991mm

Comment : CT5KN tomography with 415:1 gearbox &  512 line encoder. 
Calibrated at 5220N. Trip set to 5000N
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

file_path = r'H:\4_point_flexural\Sample3\FORCE_DISPL.xlsx'

df = pd.read_excel(file_path)

w = 5.5e-3 #m
t = 3e-3 #m
lo = 20.0e-3 #m
li = 6e-3 #m
L = 26.0e-3 #m

print(df.columns)

elong = df['Elongation'].values #mm
F = df['Force'].values #N
pos = df['Position'].values

#force/delta L
preload = (F < 1)
F_preload = F[preload]
elong_preload = elong[preload]

step1 = (F > 1) & (F < 201)
F_step1 = F[step1]
F_step1_alt= F_step1 - F_step1[0]
elong_step1 = elong[step1] 
elong_step1_alt = elong_step1 - elong_step1[0]

step2 = (F > 200) & (F < 401)
F_step2 = F[step2]
F_step2_alt= F_step2 - F_step2[0]
elong_step2 = elong[step2] 
elong_step2_alt = elong_step2 - elong_step2[0]

step3= (F > 400) & (F < 601)
F_step3 = F[step3]
F_step3_alt= F_step3 - F_step3[0]
elong_step3 = elong[step3] 
elong_step3_alt = elong_step3 - elong_step3[0]

#stress/strain
epsilon_preload = (elong_preload*1e-3)/L
epsilon_preload_alt = epsilon_preload - epsilon_preload[0]
sigma_preload = (3*F_preload*(lo - li))/(2*w*(t**2))
sigma_preload_alt = sigma_preload - sigma_preload[0]

epsilon_step1 = (elong_step1*1e-3)/L
epsilon_step1_alt = epsilon_step1 - epsilon_step1[0]
sigma_step1 = (3*F_step1*(lo - li))/(2*w*(t**2))
sigma_step1_alt = sigma_step1 - sigma_step1[0]

epsilon_step2 = (elong_step2*1e-3)/L
epsilon_step2_alt = epsilon_step2 - epsilon_step2[0]
sigma_step2 = (3*F_step2*(lo - li))/(2*w*(t**2))
sigma_step2_alt = sigma_step2 - sigma_step2[0]

epsilon_step3 = (elong_step3*1e-3)/L
epsilon_step3_alt = epsilon_step3 - epsilon_step3[0]
sigma_step3 = (3*F_step3*(lo - li))/(2*w*(t**2))
sigma_step3_alt = sigma_step3 - sigma_step3[0]

#sans preload
sans_preload = (F > 1) & (F < 601)
F_step123 = F[sans_preload]
elong_step123 = elong[sans_preload] 

epsilon_step123 = (elong_step123*1e-3)/L
sigma_step123 = (3*F_step123*(lo - li))/(2*w*(t**2))

#graph
plt.title('Elongation vs Force')
plt.xlabel(r"$\Delta$ L (mm)")
plt.ylabel("Force (N)")
plt.plot(elong[preload], F_preload, label='preload', color='black')
plt.plot(elong[step1], F_step1, label='step1 (1N-200N)', color='green')
plt.plot(elong[step2], F_step2, label='step2 (200N-400N)', color='red')
plt.plot(elong[step3], F_step3, label='step3 (400N-600N)', color='blue')
plt.legend(loc='upper left')  
plt.show()

plt.title('Elongation vs Force')
plt.xlabel(r"$\Delta$ L (mm)")
plt.ylabel("Force (N)")
plt.plot(elong_step1_alt, F_step1_alt, label='step1', color='green')
plt.plot(elong_step2_alt, F_step2_alt, label='step2', color='red')
plt.plot(elong_step3_alt, F_step3_alt, label='step3', color='blue')
plt.legend(loc='upper left')  
plt.show()

plt.title('Stress vs Strain')
plt.xlabel(r"Strain $\varepsilon$ ")
plt.ylabel(r"Stress $\sigma$ (GPa)")
plt.plot(epsilon_preload, sigma_preload*1e-9, label='preload', color='black')
plt.plot(epsilon_step1, sigma_step1*1e-9, label='step1', color='green')
plt.plot(epsilon_step2, sigma_step2*1e-9, label='step2', color='red')
plt.plot(epsilon_step3, sigma_step3*1e-9, label='step3', color='blue')
plt.legend(loc='upper left')  
plt.show()

result_123 = linregress(epsilon_step123, sigma_step123)
result_1 = linregress(epsilon_step1_alt, sigma_step1_alt)
result_2 = linregress(epsilon_step2_alt, sigma_step2_alt)
result_3 = linregress(epsilon_step3_alt, sigma_step3_alt)

plt.title(f'Stress vs Strain (E = {result_123.slope*1e-9:.2f})')
plt.xlabel(r"Strain $\varepsilon$ ")
plt.ylabel(r"Stress $\sigma$ (GPa)")
plt.plot(epsilon_step1, sigma_step1*1e-9, label='step1', color='green')
plt.plot(epsilon_step2, sigma_step2*1e-9, label='step2', color='red')
plt.plot(epsilon_step3, sigma_step3*1e-9, label='step3', color='blue')
plt.legend(loc='upper left')  
plt.show()

plt.title('Stress vs Strain')
plt.xlabel(r"Strain $\varepsilon$ ")
plt.ylabel(r"Stress $\sigma$ (GPa)")
plt.plot(epsilon_step1_alt, sigma_step1_alt*1e-9, label=f'step1 (E = {result_1.slope*1e-9:.2f})', color='green')
plt.plot(epsilon_step2_alt, sigma_step2_alt*1e-9, label=f'step2 (E = {result_2.slope*1e-9:.2f})', color='red')
plt.plot(epsilon_step3_alt, sigma_step3_alt*1e-9, label=f'step3 (E = {result_3.slope*1e-9:.2f})', color='blue')
plt.legend(loc='upper left')  
plt.show()
