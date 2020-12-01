#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:47:07 2020

@author: v1per
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

################ Experiment 80 - Pendulum #############################


########### Read CSV files

raw665_1 = np.genfromtxt('Data/Pendulum-Data/length665-1/Raw Data.csv',delimiter=',')[1:]
auto665_1 = np.genfromtxt('Data/Pendulum-Data/length665-1/Autocorrelation.csv',delimiter=',')[1:]
reso665_1 = np.genfromtxt('Data/Pendulum-Data/length665-1/Resonance.csv',delimiter=',')[1:]

raw665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Raw Data.csv',delimiter=',')[1:]
auto665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Autocorrelation.csv',delimiter=',')[1:]
reso665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Resonance.csv',delimiter=',')[1:]

raw665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Raw Data.csv',delimiter=',')[1:]
auto665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Autocorrelation.csv',delimiter=',')[1:]
reso665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Resonance.csv',delimiter=',')[1:]



############################ DATA ANALYSIS ####################################

############ CURVE FIT 

# Function from equation 2 - harmonic oscillation
def func1(t, omega, gamma, A, alpha):
    return A * np.cos(omega*t + alpha) * np.exp(-gamma*t)

# Function from equation 3 - derivative of equation 2
def func2(t, omega, gamma, A, alpha):
    return -A * (gamma*np.cos(omega*t + alpha) + omega*np.sin(omega*t + alpha)) * np.exp(-gamma*t)

def fit(time, data, func = func2): 
    
    idx = np.where(data == 0)[0][0]
    t = time[idx:]
    res = data[idx:]
    A = np.max(data)
    
    f = lambda t, omega, gamma : func(t, omega, gamma, A, 0)
    params, _ = curve_fit(f, t, res)
    
    return params, f, t, res, idx
    

########### example raw665_3 z-axis
    
# params, f, t, res, idx = fit(raw665_3[:,0], raw665_3[:,3])
# plt.figure()
# plt.plot(t, f(t, *params), 'k--', label='fit')
# # plt.plot(t, res, 'g-', label='raw')
# plt.legend()
# plt.ylabel('amplitude')
# plt.xlabel('time')
# plt.grid()
# plt.show()


################## JUST FIT TURNING POINTS TO DECAY FUNCTION

# Find turning points of the function

def find_turns(data, tol):  # Input data and Index Tolerance
    i = 1
    turns = []
    indices = []
    while(i < len(data)-tol):
        
        if abs(data[i+tol]) < abs(data[i]) and abs(data[i-tol]) < abs(data[i]):
            turns.append(abs(data[i]))
            indices.append(i)
            
        i += 1
    return np.array(turns), np.array(indices)

max665, idx665 = find_turns(raw665_3[:,3], 4)
t = np.linspace(0, 15, len(max665))
Amax665 = np.max(max665)

decay = lambda t, gamma : Amax665 * np.exp(-gamma * t)

params, _ = curve_fit(decay, t, max665)

plt.figure()
plt.plot(t, max665, 'b-')
plt.plot(t, decay(t, params), 'k--')
plt.grid()
plt.show()

print()
print('Value for Gamma is: ' + str(params[0]))
print()

# PLOT DATA


################ RAW DATA PLOTS #############################

# fp, a1 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
# a1[0].plot(raw665_1[:,0], raw665_1[:,1], 'b-', label='x')
# a1[0].legend()
# a1[0].grid()

# a1[1].plot(raw665_1[:,0], raw665_1[:,2], 'r-', label='y')
# a1[1].legend()
# a1[1].grid()

# a1[2].plot(raw665_1[:,0], raw665_1[:,3], 'g-', label='z')
# a1[2].legend()
# a1[2].grid()

# fp, a2 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
# a2[0].plot(raw665_2[:,0], raw665_2[:,1], 'b-', label='x')
# a2[0].legend()
# a2[0].grid()

# a2[1].plot(raw665_2[:,0], raw665_2[:,2], 'r-', label='y')
# a2[1].legend()
# a2[1].grid()

# a2[2].plot(raw665_2[:,0], raw665_2[:,3], 'g-', label='z')
# a2[2].legend()
# a2[2].grid()

fp, a3 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
a3[0].plot(raw665_3[:,0], raw665_3[:,1], 'b-', label='x')
a3[0].legend()
a3[0].grid()

a3[1].plot(raw665_3[:,0], raw665_3[:,2], 'r-', label='y')
a3[1].legend()
a3[1].grid()

a3[2].plot(raw665_3[:,0], raw665_3[:,3], 'g-', label='z')
# a3[2].plot(t[0], res[0], 'bo', label='origin point')
# a3[2].plot(t, f(t, *params), 'k--', label='fit')
a3[2].legend()
a3[2].grid()


############## Autocorrelation and Resonance ###################

# f1, ar = plt.subplots(2, 1, sharex=True, figsize=(6,6))

# ar[0].plot(auto665_1[:,0], auto665_1[:,1], 'k-', label='Autocorr')
# ar[0].legend()
# ar[0].grid()

# ar[1].plot(reso665_1[:,0], reso665_1[:,1], 'y-', label='Resonance')
# ar[1].legend()
# ar[1].grid()


