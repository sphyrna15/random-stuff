#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:47:07 2020

@author: v1per
"""

import numpy as np
import matplotlib.pyplot as plt

################ Experiment 80 - Pendulum #############################


# Read CSV files

raw665_1 = np.genfromtxt('Data/Pendulum-Data/length665-1/Raw Data.csv',delimiter=',')[1:]
auto665_1 = np.genfromtxt('Data/Pendulum-Data/length665-1/Autocorrelation.csv',delimiter=',')[1:]
reso665_1 = np.genfromtxt('Data/Pendulum-Data/length665-1/Resonance.csv',delimiter=',')[1:]

raw665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Raw Data.csv',delimiter=',')[1:]
auto665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Autocorrelation.csv',delimiter=',')[1:]
reso665_2 = np.genfromtxt('Data/Pendulum-Data/length665-2/Resonance.csv',delimiter=',')[1:]

raw665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Raw Data.csv',delimiter=',')[1:]
auto665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Autocorrelation.csv',delimiter=',')[1:]
reso665_3 = np.genfromtxt('Data/Pendulum-Data/length665-3/Resonance.csv',delimiter=',')[1:]




# PLOT DATA


# RAW DATA 

fp, a1 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
a1[0].plot(raw665_1[:,0], raw665_1[:,1], 'b-', label='x')
a1[0].legend()
a1[0].grid()

a1[1].plot(raw665_1[:,0], raw665_1[:,2], 'r-', label='y')
a1[1].legend()
a1[1].grid()

a1[2].plot(raw665_1[:,0], raw665_1[:,3], 'g-', label='z')
a1[2].legend()
a1[2].grid()

fp, a2 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
a2[0].plot(raw665_2[:,0], raw665_2[:,1], 'b-', label='x')
a2[0].legend()
a2[0].grid()

a2[1].plot(raw665_2[:,0], raw665_2[:,2], 'r-', label='y')
a2[1].legend()
a2[1].grid()

a2[2].plot(raw665_2[:,0], raw665_2[:,3], 'g-', label='z')
a2[2].legend()
a2[2].grid()

fp, a3 = plt.subplots(3, 1, sharex=True, figsize=(6,6))
a3[0].plot(raw665_3[:,0], raw665_3[:,1], 'b-', label='x')
a3[0].legend()
a3[0].grid()

a3[1].plot(raw665_3[:,0], raw665_3[:,2], 'r-', label='y')
a3[1].legend()
a3[1].grid()

a3[2].plot(raw665_3[:,0], raw665_3[:,3], 'g-', label='z')
a3[2].legend()
a3[2].grid()



# # Autocorrelation and Resonance 

# f1, ar = plt.subplots(2, 1, sharex=True, figsize=(6,6))

# ar[0].plot(auto665_1[:,0], auto665_1[:,1], 'k-', label='Autocorr')
# ar[0].legend()
# ar[0].grid()

# ar[1].plot(reso665_1[:,0], reso665_1[:,1], 'y-', label='Resonance')
# ar[1].legend()
# ar[1].grid()

