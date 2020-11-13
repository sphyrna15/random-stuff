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
 

radius = 202.5 / (2*np.pi) * 10**(-2)

mass_110=10**(-3)*np.array([100, 200,500,1000]) 
mass_11 = 10**(-3)*np.array([100, 100, 100, 200, 200, 200, 500, 500, 500, 1000, 1000, 1000]) 
height_11 = 10**(-2)*np.array([100,100, 90, 100, 100, 90, 100, 100, 100, 100, 100, 100])

Tn_11 = np.array([12.22, 11.94, 25.77, 17.47, 8.66, 09.03, 11.56, 
                  23.29, 17.35, 13.25, 17.62, 27.03])

n_11 = np.array([5, 5, 10, 10, 5, 5, 10, 20, 15, 15, 20, 30])


# Aufgabe 1.2 

mass_120 =10**(-3)*np.array([100, 200, 300, 500, 1000])
mass_12 =10**(-3)*np.array([100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 
                    300, 300, 500, 500, 500, 500, 1000, 1000, 1000, 1000])

height_12 = np.ones((mass_12.shape)) * 1

time_12 = np.array([2.41, 2.38, 2.35, 2.34, 1.66, 1.72, 1.66, 1.66, 1.43, 
                   1.40, 1.43, 1.35, 1.13, 1.13, 01.06, 01.09, 0.88, 
                   0.81, 0.91, 0.78])

# Aufgabe 2

mass2_ccw = 10**(-3)*np.array([500, 500, 500, 200, 200, 200, 200, 1000, 1000])
mass2_cw = 10**(-3)*np.array([1000, 1000, 1000, 100, 100, 200, 500, 500, 100, 100, 100])

n_ccw = np.array([73, 22, 76, 45, 20, 46, 43, 39, 42])
n_cw = np.array([48, 38, 29, 71, 35, 34, 55, 18, 31, 17, 20])

N_ccw = np.array([3, 3, 3, 1, 2, 1, 1, 4, 4])
N_cw = np.array([4, 4.2, 3, 1, 1, 2, 4, 3, 1, 1, 1])

time2_cw = np.array([24.13, 22.28, 16.5, 46.19, 33.34, 33.12,
                      36.31, 19.03, 29.47, 21.72, 24.84])
time2_ccw = np.array([35.56, 19.94, 36.42, 26.09, 25.00, 
                      26.50, 25.19, 21.85, 22.37])

l = 36.5*10**(-2)



####################### BERECHNUNGEN ########################################


# Aufgabe 1.1

omega = 2*np.pi / (Tn_11 / n_11)
theta1 = 2 * mass_11 * 9.81 * height_11 / omega**2 - mass_11*radius**2


# Aufgabe 1.2

theta2 = mass_12 * radius**2 * (9.81*time_12**2 / (2*height_12) - 1)


# Aufgabe 2

Omega_ccw = 2 * np.pi / (time2_ccw / N_ccw)
omega_ccw = 2 * np.pi / (time2_ccw / n_ccw)

Omega_cw = 2 * np.pi / (time2_cw / N_cw)
omega_cw = 2 * np.pi / (time2_cw / n_cw)

err_ccw1 = Omega_ccw - (mass2_ccw * 9.81 * l) / (np.average(theta1) * omega_ccw)
err_ccw2 = Omega_ccw - (mass2_ccw * 9.81 * l) / (np.average(theta2) * omega_ccw)

err_cw1 = Omega_cw - (mass2_cw * 9.81 * l) / (np.average(theta1) * omega_cw)
err_cw2 = Omega_cw - (mass2_cw * 9.81 * l) / (np.average(theta2) * omega_cw)


theta_tot = np.concatenate((theta1, theta2))

err_totcw = Omega_cw - (mass2_cw * 9.81 * l) / (np.average(theta_tot) * omega_cw)
err_totccw = Omega_ccw - (mass2_ccw * 9.81 * l) / (np.average(theta_tot) * omega_ccw)

err_tot = np.concatenate((err_totcw, err_totccw))

print('---' * 20)
print()
print('The aveagre Moment of Inertia theta1: ' + str(np.average(theta1)))
print()
print('The average Moment of Inertia theta2: ' + str(np.average(theta2)))
print()
print('The aveagre Moment of Inertia total: ' + str(np.average(theta_tot)))
print()
print('Variance of theta1: ' + str(np.var(theta1)))
print()
print('Variance of theta2: ' + str(np.var(theta2)))
print()
print('Average Value for Omega_p ccw is ' + str(np.average(Omega_ccw)))
print()
print('Average Value for Omega_p cw is ' + str(np.average(Omega_cw)))
print()
print('---' * 20)
print('Average error ccw for theta1: ' + str(np.average(err_ccw1)))
print()
print('Average error ccw for theta2: ' + str(np.average(err_ccw2)))
print()
print('Average error cw for theta1: ' + str(np.average(err_cw1)))
print()
print('Average error cw for theta2: ' + str(np.average(err_cw2)))
print()
print('Average total deviation: ' + str(np.average(err_tot)))
print()
print('Average total standard deviation: ' + str(np.std(err_tot)))
print()
print('---' * 20)



##################### GRAPHIC REPRESENTATION OF DATA ###########################

#Raw Data for Theta

topline1 = np.average(theta1) + np.std(theta1)
topline2 = np.average(theta2) + np.std(theta2)
btmline1 = np.average(theta1) - np.std(theta1)
btmline2 = np.average(theta2) - np.std(theta2)

plt.figure()

plt.plot(mass_11, theta1, 'go', label='Method 1 results for theta')
plt.plot([0,1], [np.average(theta1), np.average(theta1)], 'k--', label='Method 1 average value for theta')
plt.plot([0,1], [np.average(theta2), np.average(theta2)], 'r--', label='Method 2 average value for theta')
plt.plot([0,1], [topline1, topline1], 'b--', label='Average $\pm$ 1STD method 1')
plt.plot([0,1], [btmline1, btmline1], 'b--')

plt.xlabel('Mass in kg')
plt.ylabel('Moment of Inertia in $kgm^2$')
plt.title('Method 1 Results for Moment of Inertia')
plt.legend()
plt.grid()
plt.show()

plt.figure()

plt.plot(mass_12, theta2, 'go', label='Method 1 results for theta')
plt.plot([0,1], [np.average(theta2), np.average(theta2)], 'k--', label='Method 2 average value for theta')
plt.plot([0,1], [np.average(theta1), np.average(theta1)], 'r--', label='Method 1 average value for theta')
plt.plot([0,1], [topline2, topline2], 'b--', label='Average $\pm$ 1STD method 2')
plt.plot([0,1], [btmline2, btmline2], 'b--')

plt.xlabel('Mass in kg')
plt.ylabel('Moment of Inertia in $kgm^2$')
plt.title('Method 2 Results for Moment of Inertia')
plt.legend()
plt.grid()
plt.show()


# Raw Data for the Precession movements

param = np.polyfit(mass2_ccw, Omega_ccw, 1)
param2 = np.polyfit(mass2_cw, Omega_cw, 1)
massfit = np.arange(0, 1.1, 0.1)

plt.figure()

plt.plot(mass2_ccw, Omega_ccw, 'bo', label='Counterclockwise Precession Movement.')
plt.plot(mass2_cw, Omega_cw, 'ro', label='Clockwise Precession Movement')
plt.plot(massfit, param[0]*massfit + param[1], 'b--', label='Linear Fit of the CCW Movement')
plt.plot(massfit, param2[0]*massfit + param2[1], 'r--', label='Linear Fit of the CCW Movement')

plt.xlabel('Mass in kg')
plt.ylabel('Angular Precession Velocity in ...')
plt.title('Precession Velocity in Relation to Mass')
plt.grid()
plt.legend()
plt.show()


