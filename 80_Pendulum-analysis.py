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

raw665 = np.genfromtxt('Data/Pendulum-Data/length665-1/Raw Data.csv',delimiter=',')[1:]

auto665 = np.genfromtxt('Data/Pendulum-Data/length665-1/Autocorrelation.csv',delimiter=',')[1:]

reso665 = np.genfromtxt('Data/Pendulum-Data/length665-1/Resonance.csv',delimiter=',')[1:]








# PLOT DATA


f1, a1 = plt.subplots(2, 1, sharex=True, figsize=(6,6))

a1[0].plot(auto665[:,0], auto665[:,1], 'k-', label='Autocorr')
a1[0].legend()
a1[0].grid()

a1[1].plot(reso665[:,0], reso665[:,1], 'y-', label='Resonance')
a1[1].legend()
a1[1].grid()

fp, ap = plt.subplots(3, 1, sharex=True, figsize=(6,6))
ap[0].plot(raw665[:,0], raw665[:,1], 'b-', label='x')
ap[0].legend()
ap[0].grid()

ap[1].plot(raw665[:,0], raw665[:,2], 'r-', label='y')
ap[1].legend()
ap[1].grid()

ap[2].plot(raw665[:,0], raw665[:,3], 'g-', label='z')
ap[2].legend()
ap[2].grid()

