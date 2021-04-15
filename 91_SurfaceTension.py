#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:24:30 2021

"""

import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy



""" DATA """

# Olive Oil
oil_Ds = np.array([24, 25, 27, 26, 25, 26, 25, 27, 26, 25])
oil_De = np.array([28, 29, 32, 31, 28, 20, 30, 31, 29, 28])
d_oil = np.array([42, 43, 44, 42, 41, 42, 43, 44, 43, 41])

# Water
water_Ds = np.array([38, 33, 32, 32, 31, 21, 32, 29, 35, 40])
water_De = np.array([45, 44, 44, 45, 44, 29, 46, 43, 53, 55])
d_water = np.array([56, 55, 53, 54, 54, 37, 55, 51, 60, 66])

""" Constants """

d = 0.0045 # tube diameter in m (4.5mm)


g = 9.81
a = 0.345
b = -2.5
rho_water = 1000 # units kg/m^3
rho_air = 1.225 # units kg/m^3 at 15 degrees
rho_oil = 1395 # units kg/m^3



""" Results """

#convert units
wat_Ds = (d/d_water) * water_Ds # units [m]
wat_De = (d/d_water) * water_De #units [m]
o_Ds = (d/d_oil) * oil_Ds #units [m]
o_De = (d/d_oil) * oil_De #units [m]

gamma_water = a*(rho_water-rho_air)*g * (wat_De**2)*((wat_Ds/wat_De)**b)
gamma_oil = a*(rho_oil-rho_air)*g * (o_De**2) * ((o_Ds/o_De)**b) 

#average
avg_water = np.average(gamma_water)
avg_oil = np.average(gamma_oil)
#standard deviation
std_water = np.std(gamma_water)
std_oil = np.std(gamma_oil)


print('---'*25)
print("Avg Surface Tension of water is {} N/m".format(avg_water))
print()
print("Avg Surface Tension of Olive Oil is {} N/m".format(avg_oil))
print("---"*25)

print('---'*25)
print("We find standard deviations: ")
print("Water - {}".format(std_water))
print("Olive Oil - {}".format(std_oil))
print("---"*25)

lit_water = 7.28e-2
lit_oil = 3.3e-2
print('---'*25)
print("Literature Values: ")
print("Water - {} N/m".format(lit_water))
print("Olive Oild - {} N/m".format(lit_oil))
print("---"*25)



""" Calculating with uncertainties """

du = ufloat(0.0045, 0.0005)

wat_De_avg = np.average(wat_De)
wat_Ds_avg = np.average(wat_Ds)
wat_De_std = np.std(wat_De)
wat_Ds_std = np.std(wat_Ds)

watu_De = ufloat(wat_De_avg, wat_De_std) #with uncertainties
watu_Ds = ufloat(wat_Ds_avg, wat_Ds_std)

oil_De_avg = np.average(o_De)
oil_Ds_avg = np.average(o_Ds)
oil_De_std = np.std(o_De)
oil_Ds_std = np.std(o_Ds)

oilu_De = ufloat(oil_De_avg, oil_De_std) #with uncertainties
oilu_Ds = ufloat(oil_Ds_avg, oil_Ds_std)

gammau_water = a*(rho_water-rho_air)*g * (watu_De**2)*((watu_Ds/watu_De)**b)
gammau_oil = a*(rho_oil-rho_air)*g * (oilu_De**2) * ((oilu_Ds/oilu_De)**b) 


print('---'*25)
print("Surface Tension of Water is {} N/m".format(gammau_water))
print()
print("Surface Tension of Olive Oil is {} N/m".format(gammau_oil))
print("---"*25)



 
 



