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
# auto665_1 = np.genfromtxt('Data/Pendulum-Data/length665-1/Autocorrelation.csv',delimiter=',')[1:]
# reso665_1 = np.genfromtxt('Data/Pendulum-Data/length665-1/Resonance.csv',delimiter=',')[1:]
raw665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Raw Data.csv',delimiter=',')[1:]
# auto665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Autocorrelation.csv',delimiter=',')[1:]
# reso665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Resonance.csv',delimiter=',')[1:]
raw665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Raw Data.csv',delimiter=',')[1:]
# auto665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Autocorrelation.csv',delimiter=',')[1:]
# reso665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Resonance.csv',delimiter=',')[1:]

raw690_1 = np.genfromtxt('Data/Pendulum-Data/length690-1/Raw Data.csv',delimiter=',')[1:]
raw690_2 = np.genfromtxt('Data/Pendulum-Data/length690-2/Raw Data.csv',delimiter=',')[1:]
raw690_3 = np.genfromtxt('Data/Pendulum-Data/length690-3/Raw Data.csv',delimiter=',')[1:]

raw760_1 = np.genfromtxt('Data/Pendulum-Data/length760-1/Raw Data.csv',delimiter=',')[1:]
raw760_2 = np.genfromtxt('Data/Pendulum-Data/length760-2/Raw Data.csv',delimiter=',')[1:]
raw760_3 = np.genfromtxt('Data/Pendulum-Data/length760-3/Raw Data.csv',delimiter=',')[1:]

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

# Get the time values for the turning points from the indices
def get_times(time_data, indices):
    
    times = []
    for idx in indices:
        times.append(time_data[idx])
    
    return np.array(times)


# Function to fully calculate the decay coeff. gamma and plot the decay
def calculate_gamma(data, time_data, tol, plot = True, title = 'Title'):
    
    extrema, indices = find_turns(data, tol)
    time = get_times(time_data, indices) # Not sure yet how to structure time
    t0 = time[0] ; t = time - t0
    # t = np.linspace(0, 10, len(extrema))
    Amax = np.max(extrema)
    
    decay = lambda t, gamma : Amax * np.exp(-gamma * t)
    
    gamma_value, _ = curve_fit(decay, t, extrema)
    
    if plot != True: 
        return gamma_value[0], extrema, indices, time
    
    plt.figure()
    plt.plot(t, extrema, 'b-', label='Extrema')
    plt.plot(t, decay(t, gamma_value[0]), 'k--', label='Fitted Decay Function')
    plt.title(title)
    plt.xlabel('Time $t$ (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
    
    return gamma_value[0], extrema, indices, time


########### Calculate Gammas 

gamma665_1, ex765_1, idx665_1, t665_1 = calculate_gamma(raw665_1[:,1], raw665_1[:,0], 10, 
                                                        title = 'L-665 1', plot=False)

gamma665_2, ex765_2, idx665_2, t665_2 = calculate_gamma(raw665_2[:,1], raw665_2[:,0], 10, 
                                                        title = 'L-665 2', plot=False)

gamma665_3, ex665_3, idx665_3, t665_3 = calculate_gamma(raw665_3[:,3], raw665_3[:,0], 10,
                                                        title = 'L-690 3', plot=False)

gamma760_1, ex760_1, idx760_1, t760_1 = calculate_gamma(raw760_1[:,1], raw760_1[:,0], 10, 
                                                        title = 'L-760 1', plot=False)

gamma760_2, ex760_2, idx760_2, t760_2 = calculate_gamma(raw760_2[:,1], raw760_2[:,0], 10, 
                                                        title = 'L-760 2', plot=False)

gamma760_3, ex760_3, idx760_3, t760_3 = calculate_gamma(raw760_3[:,1], raw760_3[:,0], 10, 
                                                        title = 'L-760 3', plot=False)

gamma690_1, ex760_1, idx690_1, t690_1 = calculate_gamma(raw690_1[:,1], raw690_1[:,0], 10, 
                                                        title = 'L-690 1', plot=False)

gamma690_2, ex760_2, idx690_2, t690_2 = calculate_gamma(raw690_2[:,1], raw690_2[:,0], 10, 
                                                        title = 'L-690 2', plot=False)

gamma690_3, ex760_3, idx690_3, t690_3 = calculate_gamma(raw690_3[:,1], raw690_3[:,0], 10, 
                                                        title = 'L-690 3', plot=False)

# Print Decay coefficients
print()
print('Gamma for L-665 1: ' + str(gamma665_1))
print('Gamma for L-665 2: ' + str(gamma665_2))
print('Gamma for L-665 3: ' + str(gamma665_3))
print()
print('Gamma for L-760 1: ' + str(gamma760_1))
print('Gamma for L-760 2: ' + str(gamma760_2))
print('Gamma for L-760 3: ' + str(gamma760_3))
print()
print('Gamma for L-690 1: ' + str(gamma690_1))
print('Gamma for L-690 2: ' + str(gamma690_2))
print('Gamma for L-690 3: ' + str(gamma690_3))
print()


################ RAW DATA PLOTS #############################

############## Length 760 1 and 2

# fp, a1 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
# a1[0].plot(raw760_1[:,0], raw760_1[:,1], 'b-', label='x')
# a1[0].plot(t760_1, get_times(raw760_1[:,1], idx760_1), 'bo', label='extrema')
# a1[0].legend()
# a1[0].grid()

# a1[1].plot(raw760_1[:,0], raw760_1[:,2], 'r-', label='y')
# a1[1].legend()
# a1[1].grid()

# a1[2].plot(raw760_1[:,0], raw760_1[:,3], 'g-', label='z')
# a1[2].legend()
# a1[2].grid()

# fp, a2 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
# a2[0].plot(raw760_2[:,0], raw760_2[:,1], 'b-', label='x')
# a2[0].plot(t760_2, get_times(raw760_2[:,1], idx760_2), 'bo', label='extrema')
# a2[0].legend()
# a2[0].grid()

# a2[1].plot(raw760_2[:,0], raw760_2[:,2], 'r-', label='y')
# a2[1].legend()
# a2[1].grid()

# a2[2].plot(raw760_2[:,0], raw760_2[:,3], 'g-', label='z')
# a2[2].legend()
# a2[2].grid()


########### Length 665 1

# fp, a3 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
# a3[0].plot(raw665_3[:,0], raw665_3[:,1], 'b-', label='x')
# a3[0].legend()
# a3[0].grid()

# a3[1].plot(raw665_3[:,0], raw665_3[:,2], 'r-', label='y')
# a3[1].legend()
# a3[1].grid()

# a3[2].plot(raw665_3[:,0], raw665_3[:,3], 'g-', label='z')
# plt.plot(t665, get_times(raw665_3[:,3], idx665), 'bo', label='extrema')
# a3[2].legend()
# a3[2].grid()


########### Length 690 1, 2 and 3

# fp, a4 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
# a4[0].plot(raw690_3[:,0], raw690_3[:,1], 'b-', label='x')
# a4[0].legend()
# a4[0].grid()

# a4[1].plot(raw690_3[:,0], raw690_3[:,2], 'r-', label='y')
# a4[1].legend()
# a4[1].grid()

# a4[2].plot(raw690_3[:,0], raw690_3[:,3], 'g-', label='z')
# a4[2].legend()
# a4[2].grid()

# fp, a5 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
# a5[0].plot(raw690_2[:,0], raw690_2[:,1], 'b-', label='x')
# a5[0].legend()
# a5[0].grid()

# a5[1].plot(raw690_2[:,0], raw690_2[:,2], 'r-', label='y')
# a5[1].legend()
# a5[1].grid()

# a5[2].plot(raw690_2[:,0], raw690_2[:,3], 'g-', label='z')
# a5[2].legend()
# a5[2].grid()

# fp, a6 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
# a6[0].plot(raw690_1[:,0], raw690_1[:,1], 'b-', label='x')
# a6[0].legend()
# a6[0].grid()

# a6[1].plot(raw690_1[:,0], raw690_1[:,2], 'r-', label='y')
# a6[1].legend()
# a6[1].grid()

# a6[2].plot(raw690_1[:,0], raw690_1[:,3], 'g-', label='z')
# a6[2].legend()
# a6[2].grid()


############## Autocorrelation and Resonance ###################

# f1, ar = plt.subplots(2, 1, sharex=True, figsize=(6,6))

# ar[0].plot(auto665_1[:,0], auto665_1[:,1], 'k-', label='Autocorr')
# ar[0].legend()
# ar[0].grid()

# ar[1].plot(reso665_1[:,0], reso665_1[:,1], 'y-', label='Resonance')
# ar[1].legend()
# ar[1].grid()


