#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:35:44 2020

@author: v1per
"""

import numpy as np
import matplotlib.pyplot as plt

""" Experiment Kreisel Data Analysis """


################################### IMPORT DATA #####################################



# Aufgabe 1.1 

radius = 202.5 / 2*np.pi

mass_11 = np.array([100, 100, 100, 200, 200, 200, 500, 500, 500, 1000, 1000, 1000]) 
height_11 = np.array([100,100, 90, 100, 100, 90, 100, 100, 100, 100, 100, 100])

Tn_11 = np.array([12.22, 11.94, 25.77, 17.47, 8.66, 09.03, 11.56, 
                  23.29, 17.35, 13.25, 17.62, 27.03])

n_11 = np.array([5, 5, 10, 10, 5, 5, 10, 20, 15, 15, 20, 30])


# Aufgabe 2.1 

mass_12 = np.array([100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 
                    300, 300, 500, 500, 500, 500, 1000, 1000, 1000, 1000])

height_12 = np.ones((mass_12.shape)) * 100

time_12 = np.array([2.41, 2.38, 2.35, 2.34, 1.66, 1.72, 1.66, 1.66, 1.43, 
                   1.40, 1.43, 1.35, 1.13, 1.13, 01.06, 01.09, 0.88, 
                   0.81, 0.91, 0.78])

# Aufgabe 2

mass2_ccw = np.array([500, 500, 500, 200, 200, 200, 200, 1000, 1000])
mass2_cw = np.array([1000, 1000, 1000, 100, 100, 200, 500, 500, 100, 100, 100])

n_ccw = np.array([73, 22, 76, 45, 20, 46, 43, 39, 42])
n_cw = np.array([48, 38, 29, 71, 35, 34, 55, 18, 31, 17, 20])

N_ccw = np.array([3, 3, 3, 1, 2, 1, 1, 4, 4])
N_cw = np.array([4, 4.2, 3, 1, 1, 2, 4, 3, 1, 1, 1])

time2_cw = np.array([24.13, 22.28, 16.5, 46.19, 33.34, 33.12,
                      36.31, 19.03, 29.47, 21.72, 24.84])
time2_ccw = np.array([35.56, 19.94, 36.42, 26.09, 25.00, 
                      26.50, 25.19, 21.85, 22.37])

l = 36.5



####################### BERECHNUNGEN ########################################


# Aufgabe 1.1

omega = 2*np.pi / (Tn_11 / n_11)
theta1 = 2 * mass_11 * 9.81 * height_11 / omega**2 - mass_11*radius**2


# Aufgabe 1.2

theta2 = mass_12 * radius**2 * (9.81*time_12**2 / (2*height_12) - 1)


# Aufgabe 2

Omega_ccw = 2 * np.pi / (time2_ccw / N_ccw)
omega_ccw = 2 * np.pi / (time2_ccw / N_ccw)

Omega_cw = 2 * np.pi / (time2_cw / N_cw)
omega_cw = 2 * np.pi / (time2_cw / N_cw)

err_ccw1 = Omega_ccw - (mass2_ccw * l) / (np.average(theta1) * omega_ccw)
err_ccw2 = Omega_ccw - (mass2_ccw * l) / (np.average(theta2) * omega_ccw) 

err_cw1 = Omega_cw - (mass2_cw * l) / (np.average(theta1) * omega_cw)
err_cw2 = Omega_cw - (mass2_cw * l) / (np.average(theta2) * omega_cw) 

print('---' * 20)
print()
print('Average error ccw for theta1: ' + str(np.average(err_ccw1)))
print()
print('Average error ccw for theta2: ' + str(np.average(err_ccw2)))
print()
print('Average error cw for theta1: ' + str(np.average(err_cw1)))
print()
print('Average error cw for theta2: ' + str(np.average(err_cw2)))
print()
print('---' * 20)


