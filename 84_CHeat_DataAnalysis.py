#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:14:18 2020

@author: v1per
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


#Get the start and end Time of the Power applied Indices

idxwat = np.where(Pwat > 2.0)[0]
idxalu = np.where(Palu > 2.0)[0]
idxcop= np.where(Pcop > 2.0)[0]

dtw = (idxwat[-1] - idxwat[0]) / 2.5
dta = idxalu[-1] - idxalu[0]
dtc = idxcop[-1] - idxcop[0]

dQwater_avg = np.average(Pwat[idxwat[0]:idxwat[-1]])*dtw
dQwat = np.trapz(Pwat[idxwat[0]:idxwat[-1]], x = twat[idxwat[0]:idxwat[-1]])

print()
print('Average Heat dQwat: ' +str(dQwater_avg))
print('Integral actual Heat dQwat: ' + str(dQwat))

dQalu = np.trapz(Palu[idxalu[0]:idxalu[-1]], x = t[idxalu[0]:idxalu[-1]])
dQcop = np.trapz(Pcop[idxcop[0]:idxcop[-1]], x = t[idxcop[0]:idxcop[-1]])

print('Integral actual Heat dQalu: ' + str(dQalu))
print('Integral actual Heat dQcop: ' + str(dQcop))


#Calculate Integrals for Temperature dependance


T1wat = Twat[idxwat[0]]
DTwat = Twat - T1wat

T1alu = Talu[idxalu[0]]
DTalu = Talu - T1alu

T1cop = Tcop[idxcop[0]]
DTcop = Tcop - T1cop

Fwat = np.trapz(DTwat[idxwat[0]:idxwat[-1]], x = twat[idxwat[0]:idxwat[-1]])
Falu = np.trapz(DTalu[idxalu[0]:idxalu[-1]], x = t[idxalu[0]:idxalu[-1]])
Fcop = np.trapz(DTcop[idxcop[0]:idxcop[-1]], x = t[idxcop[0]:idxcop[-1]])

print()
print('Integral Fwat: ' +str(Fwat))
print('Integral Falu: ' +str(Falu))
print('Integral Fcop: ' +str(Fcop))

# Calculate derivatives Tdot

dtidx = 100 #Index fit range

f = lambda x, a, b : a*x + b
params_wat, _ = curve_fit(f, twat[idxwat[-1]:], 
                          DTwat[idxwat[-1]:])

params_alu, _ = curve_fit(f, t[idxalu[-1]:], 
                          DTalu[idxalu[-1]:])

params_cop, _ = curve_fit(f, t[idxcop[-1]:], 
                          DTcop[idxcop[-1]:])

paramcop = np.polyfit(t[idxcop[0]:idxcop[-1]], DTcop[idxcop[0]:idxcop[-1]],deg=1)


plt.figure()
plt.plot(t, DTcop, color='tab:green', label = 'Temperature')
plt.plot(t, Pcop, 'r-', label='Power')
plt.plot(t[idxcop[0]], DTcop[idxcop[0]], 'bo')
plt.plot(t[idxcop[-1]], DTcop[idxcop[0-1]], 'ko')
plt.plot(t, f(t, params_cop[0], params_cop[1]), 'k--', label='Linear Fit to Temperature loss')
plt.title('Time Developement to Temperatuer and Power')
plt.xlabel('Time')
plt.ylabel('Power P and Temperature T')
plt.legend()
plt.grid()



Tdotwat = params_wat[0]
Tdotalu = params_alu[0]
Tdotcop = params_cop[0]

print()
print('Gradient Tdotwat: ' + str(Tdotwat))
print('Gradient Tdotalu: ' + str(Tdotalu))
print('Gradient Tdotcop: ' + str(Tdotcop))


# NOW EVALUATE FORMULA

dTwat = Twat[idxwat[-1]] - Twat[idxwat[0]]
dTalu = Talu[idxalu[-1]] - Talu[idxalu[0]]
dTcop = Tcop[idxcop[-1]] - Tcop[idxcop[0]]

# Vshwat = np.average(Vres_wat[idxwat[0]:idxwat[-1]])
# Vwat = np.average(V_wat[idxwat[0]:idxwat[-1]])

# Vshalu = np.average(Vres_alu[idxalu[0]:idxalu[-1]])
# Valu = np.average(V_alu[idxalu[0]:idxalu[-1]])

# Vshcop = np.average(Vres_cop[idxcop[0]:idxcop[-1]])
# Vcop = np.average(V_cop[idxcop[0]:idxcop[-1]])

Ctwat = (dQwat * dTwat) / (dTwat**2 - Tdotwat*Fwat)

Ctalu = (dQalu * dTalu) / (dTalu**2 - Tdotalu*Falu)  -  Ctwat
Ctcop = (dQcop * dTcop) / (dTcop**2 - Tdotcop*Falu)  -  Ctwat

CMcop = Ctcop * (63.546/1500)
CMalu = Ctalu * (26.9815395/475)

print()
print('Ctot for water: ' + str(Ctwat))
print('Ctot for aluminium: ' + str(Ctalu))
print('Ctot for copper: ' + str(Ctcop))

print()
print('CM for aluminium: ' + str(CMalu))
print('CM for copper: ' + str(CMcop))



