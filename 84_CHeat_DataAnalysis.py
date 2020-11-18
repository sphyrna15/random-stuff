#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:14:18 2020

@author: v1per
"""

import numpy as np
import matplotlib.pyplot as plt

################### 84 SPECIFIC HEAT DATA ANALYSIS #################################

#Import Data from /Data folder .txt files

watertxt = np.loadtxt(r'Data/water.txt')
coppertxt = np.loadtxt(r'Data/aluminium.txt')
aluminiumtxt = np.loadtxt(r'Data/copper.txt')

V_wat = watertxt[:,0] * 10 #10x since a voltage divider reduces V prior to measurement
Vres_wat = watertxt[:,1]
Vbr_wat = watertxt[:,2]

V_cop = coppertxt[:,0] * 10 #10x since a voltage divider reduces V prior to measurement
Vres_cop = coppertxt[:,1]
Vbr_cop = coppertxt[:,2]

V_alu = aluminiumtxt[:,0] * 10 #10x since a voltage divider reduces V prior to measurement
Vres_alu = aluminiumtxt[:,1]
Vbr_alu = aluminiumtxt[:,2]


# Calculate time vectors for the measurements

t = np.linspace(1, len(aluminiumtxt[:,0]), len(aluminiumtxt[:,0])) # 1 measurement per second
twat = t / 2.5 #2.5 scans per second for water

print(f"Measurement time water: {twat[-1]/60:3.1f} min") # twater[-1] returns the last time point in seconds
print(f"Measurement time aluminium/copper: {t[-1]/60:3.1f} min")

f,a=plt.subplots(3,1,sharex=True,figsize=(6,6))
a[0].plot(V_wat,label='water')
a[0].plot(V_alu,label='aluminium')
a[0].plot(V_cop,label='copper')
a[0].set(ylabel='$V$ in Volt')
a[0].grid()
a[0].legend()

a[1].plot(Vres_wat,label='water')
a[1].plot(Vres_alu,label='aluminium')
a[1].plot(Vres_cop,label='copper')
a[1].set(ylabel='$V_{Sh}$ in Volt')
a[1].legend()
a[1].grid()

a[2].plot(Vbr_wat,label='water')
a[2].plot(Vbr_alu,label='aluminium')
a[2].plot(Vbr_cop,label='copper')
a[2].set(xlabel='Data Points',ylabel='$V_{Br}$ in Volt')
a[2].legend()
a[2].grid()


#Calculating the Power applied to the sample

#heating current is I = V_sh / R_sh
# Power is heating current times Voltage applied P = IV
Pwat = Vres_wat * V_wat
Palu = Vres_alu * V_alu
Pcop = Vres_cop * V_cop


# Calculating the temperature from the bridge voltage measurements

P = lambda x : 24.73 + 59.81 * x + 4.26 * x**2 #Calibration curve polynomial

Twat = P(Vbr_wat)
Talu = P(Vbr_alu)
Tcop = P(Vbr_cop)


# Plot power applied and temperature plot

fp, ap = plt.subplots(2, 1, sharex=True, figsize=(6,6))
ap[0].plot(twat,Pwat,label='water')
ap[0].plot(t,Palu,label='aluminium')
ap[0].plot(t,Pcop,label='copper')
ap[0].set(ylabel='Power (W)')
ap[0].legend()
ap[0].grid()

ap[1].plot(twat, Twat, label='water')
ap[1].plot(t, Talu, label = 'aluminuim')
ap[1].plot(t, Tcop, label = 'copper')
ap[1].set(xlabel = 'Time in seconds', ylabel='Temperature in $^o$C')
ap[1].legend()
ap[1].grid()


