#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:45:20 2020

@author: v1per
"""

import numpy as np
import matplotlib.pyplot as plt

""" Template for creating plots of lab data """


############### BETASTRAHLER ##############################


# AUFGABE 1

# Import Data

potential = np.array([ 400, 500, 600, 700, 800, 900, 1000])

time = 20

N = np.array([1428, 1560, 1606, 1625, 1590, 1590, 1584])



# plot


# plt.figure()
# plt.plot(potential, N, '-o', label='Detecteded decay events')
# plt.xlabel('Potential V')
# plt.ylabel('Number of decay events $N(V)$')
# plt.title('Detected decay events at different potentials during 20 second intervalls')
# plt.legend()
# plt.grid()
# plt.show()


# AUFGABE 2

# Import Data

t2 = np.array([30, 30, 30, 40, 45, 50, 50, 55, 55, 60, 65, 70])
N2 = np.array([2330, 1508, 1100, 760, 402, 161, 92, 60, 49, 40, 38, 34])
x = np.array([0.00, 0.25, 0.50, 1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00])
solid_angle = 0.0016

activity = (N2 / t2) / solid_angle

kBq = activity / 1000


# Plot

# plt.figure()
# plt.plot(x, kBq, '-o', label='Activity in $[kBq]$')
# plt.xlabel('Absorber sheet thickness in cm')
# plt.ylabel('Calculated activity')
# plt.title('Activity of the $^{90}Sr$-source with absorber sheets')
# plt.legend()
# plt.grid()
# plt.show()


######################### Refractive Index Prism ########################

# Refraction andle phi
betas = [238.6, 239.85, 236.39]
beta = np.average(betas)
phi = beta - 180
print('refraction angle phi = ' + str(phi))

colors = np.array(['red', 'yellow', 'green', 'bluegreen', 'indigo', 'purple1', 'purple2'])
wavelength = np.array([623.4, 579.1, 546.1, 491.6, 435.8, 407.8, 404.7])
A1 = np.array([19.0, 19.35, 19.80, 21.1, 22.75,  23.66, 23.85])
A2 = np.array([276.5, 275.5, 274.98, 273.0, 271.25, 269.5, 269.0])

delta = 0.5*((360.0 - A2) + A1)
delta_rad = np.radians(delta)
phi_rad = np.radians(phi)

ns = np.sin((delta_rad + phi_rad)/2) / np.sin(phi_rad/2)
nu = 2.99e8 / wavelength

f = lambda x : 1 / (x**2 - 1)

fns = f(ns)

model, residuals, _, _, _ = np.polyfit(nu**2, f(ns), 1, full=True)

plt.figure()

for i in range(len(colors)):
    plt.text(nu[i]**2, f(ns)[i], '  ' + colors[i])

plt.plot(nu**2, f(ns), 'o', label=r'$f(n(\nu))$')
plt.plot(nu**2, model[0]*nu**2 + model[1], 'k--', label='LeastSquares Solution')
plt.xlabel(r'Frequencies squared $\nu^2$')
plt.ylabel('Function of refractive indices $n$')
plt.title('Plotting the Dispersion formula')
plt.legend()
plt.grid()
plt.show()












