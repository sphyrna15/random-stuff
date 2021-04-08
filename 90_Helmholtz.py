#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:00:44 2021

"""

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy

###########################
# Speed of sound helmholz
###########################

# Bottle Dimensions with uncertainties
Lu = ufloat(0.065, 0.002)
Du = ufloat(0.034, 0.001)
# Bottle dimensions without uncertainties
L = 0.065
D = 0.034

mass = np.array([39, 62, 90, 146, 170, 204, 225, 234, 247, 259, 274, 290]) #in gram

freq = np.array([251.61, 269.53, 281.25, 291.43 , 304.68, 320.73, 
                  328.13, 339.70, 345.84, 351.56, 363.28, 381.0]) #in Hz

def getVolume(mass, ref):
    #reference mass filled up to bottle neck ref
    m = np.abs(mass - ref)
    # 1'000'000 g water = 1m^3 water volume
    vol = m / 1000000
    
    return vol

volume = getVolume(mass, 499)

# ################# REX DATA ##############################
# Lu = ufloat(0.09, 0.001)
# Du = ufloat(0.02, 0.001)
# freq = np.array([117.19, 140.62, 140.62, 145, 155, 164, 164, 175, 190, 215, 250])
# volume = np.array([0.00065, 0.000600, 0.000550, 0.000500, 0.000450,0.000400, 
#                     0.000350, 0.000300, 0.000250, 0.000200, 0.000150])
# #########################################################

V = 1 / volume 

print("---"*15)
print("Number of Measurements: "+str(freq.shape[0]))
print("---"*15)

c = 1.46212 #Constant for speed of sound
expu = (343**2)*(Du**2) / (16*np.pi*(Lu + c*Du/2)) #Expected Slope

exp = (343**2)*(D**2) / (16*np.pi*(L + c*D/2)) #Expected Slope

print("---"*15)
print("Expected Slope: "+str(expu))
print("---"*15)

#Linear Fit
model = np.polyfit(V, freq**2, 1, cov=True)
s = model[0][0] #SLOPE
const = model[0][1]

#uncertainties
cov = model[1]
s_err = np.sqrt(cov[0][0])

print("Measured Slope: {} +/- {}".format(round(s,4), round(s_err,4)))
print("---"*15)

plt.figure()
plt.plot(V, freq**2, 'ro', label="Data Points")
plt.plot(V, V*s + const, 'k--', label="Linear Fit")
# plt.plot(V, V*exp + const, 'g--', label="Expected Slope")
plt.ylabel("Resonance Frequency $f^2$")
plt.xlabel("per Volume $1/V$")
plt.grid()
plt.legend()
plt.show()


### Speed Of Sound

v = np.sqrt(s * (16*np.pi*(L + c*D/2)) / (D**2) )

# With Uncertainties
su = ufloat(s, s_err)
vu = unumpy.sqrt(su * (16*np.pi*(Lu + c*Du/2)) / (Du**2) )

print("---"*15)
print("Speed of Sound: "+str(vu))
print("---"*15)


import scipy.stats as stats

a, b = 245, 258
mu, sigma = 251.61, 5.24
dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

values = dist.rvs(5)





