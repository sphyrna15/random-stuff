#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:59:40 2021

"""
import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
# =============================================================================
# Part 1
# =============================================================================

directory = "Data/70_ThermalNoise/"

# CONSTANTS
G = 1e7 #V^2/m , detection gain
m = 5.5e-15 #kg, resonantor mass
kb = 1.38e-23

def getTemperature(filename, pr = True): #pr print variable
    
    directory = "Data/70_ThermalNoise/"
    data = pd.read_table(directory + filename, sep=",", usecols=['% frequency (Hz)', 'PSD (V^2/Hz)'])
    freq = data['% frequency (Hz)'].values #frequencies
    PSD = data['PSD (V^2/Hz)'].values / G**2 #displacement PSD 
    sigma = np.trapz(PSD, freq)  #PARSEVAL integral to find (\sigma_x)^2
    
    #check that resulting temperature is accurate (300K)
    res_freq = freq[np.argmax(PSD)] # resonance frequency
    omega = 2*np.pi * res_freq
    T = m*(omega**2) * sigma/kb #temperature with equipartition theorem
    if pr:
        
        print('---'*25)
        print("Found T = {}K withouh amplifier noise - should be 300K".format(round(T,3)))
        print('---'*25)
        
    return T, PSD, freq, sigma, res_freq, data

# IMPORT DATA
filename1 = "/PSD-T-300K-Srate-15000Hz-tmeas-30-ampnoise-false.dat" #task 1 
T1, PSD1, freq1, sig21, res_freq2, data2 = getTemperature(filename1)


plt.figure()
plt.loglog(freq1, PSD1, label = "PSD-$T=300$K-sRate$=15$kHz-$t=30$s-ampnoise-false")
plt.xlabel("$f$ (Hz)")
plt.ylabel("$S_x$ $(m^2/Hz)$")
plt.title("Displacement PSD vs sampled Frequencies")
plt.legend()
plt.grid()
plt.show()


# =============================================================================
# Part 2 - varying sampling rate and measurement time
# =============================================================================
#%%

files = os.listdir(directory + "/VarySampleRate")
sRate = np.array([60, 45, 30, 15])
T300 = np.ones_like(sRate) * 300

Temp = []
for file in files:
    T = getTemperature("VarySampleRate/"+file, pr=False)
    Temp.append(T[0])
    
# plt.figure()
# plt.plot(sRate, Temp, 'bx')
# plt.plot(sRate, T300, 'k--')
# plt.xlabel("Measurement Time")
# plt.ylabel("Temperature in K")
# plt.grid()
# plt.show()
    

# =============================================================================
# Part 3 - now we add amplifier noise
# =============================================================================

#Additional Noise
def getNoisyTemp(filename, pr = True):
    
    directory = "Data/70_ThermalNoise/"
    data = pd.read_table(directory + filename, sep=",", usecols=['% frequency (Hz)', 'PSD (V^2/Hz)'])
    freq = data['% frequency (Hz)'].values #frequencies
    PSD = data['PSD (V^2/Hz)'].values / G**2 #displacement PSD  
    
    #array position of resonance frequency
    res_freq_pos = np.argmax(PSD)
    #resonance frequency
    res_freq = freq[np.argmax(PSD)]
    
    #average noise measured over 2/3 the distance from 0 to the resonance frequency
    avg_noise = sum((PSD[:int(res_freq_pos*2/3)])) / len((PSD[:int(res_freq_pos*2/3)]))
    
    #calcualtion of sigma^2 using Parseval Theorem
    sig2_corr = np.trapz(PSD - avg_noise, freq) #subtracting the noise
    sig2_uncorr = np.trapz(PSD, freq) #not subtracting the noise
    #calculation of omega
    omega = 2*np.pi*res_freq
    
    #calculation of T using the equipartition theorem
    T_corr = m*omega**2*sig2_corr/kb
    T_uncorr = m*omega**2*sig2_uncorr/kb
    
    if pr:
        print('---'*25)
        print("Corrected Temperature {}K vs. noisy Temperature {}K".format(round(T_corr,3), round(T_uncorr,3)))
        print('---'*25)
        print()
    
    return T_corr, T_uncorr, PSD, freq

filename3 = "PSD-T-300K-Srate-15000Hz-tmeas-90-ampnoise-true.dat"

T3, T_noisy3, PSD3, freq3 = getNoisyTemp(filename3)

plt.figure()
plt.loglog(freq3, PSD3, label = "PSD-$T=300$K-sRate$=15$kHz-$t=90$s-ampnoise-true")
plt.xlabel("$f$ (Hz)")
plt.ylabel("$V_{PSD}$ $(V^2/Hz)$")
plt.title("Displacement PSD vs sampled Frequencies With Amplifier Noise")
plt.legend()
plt.grid()
plt.show()


# =============================================================================
# Part 4 - experiments at different Temperatures
# =============================================================================

""" 1000 KELVIN """
files = os.listdir(directory + "/1000Kelvin")
t_1000 = []
for file in files:
    T, _, _, _ = getNoisyTemp("1000Kelvin/" + file, pr=False)
    t_1000.append(T)

t_1000 = np.array(t_1000)
mean_1000 = np.mean(t_1000)
stddev_1000 = np.std(t_1000)
stderr_1000 = sem(t_1000)

""" 300 KELVIN """
files = os.listdir(directory + "/300Kelvin")
t_300 = []
for file in files:
    T, _, _, _ = getNoisyTemp("300Kelvin/" + file, pr=False)
    t_300.append(T)

t_300 = np.array(t_300)
mean_300 = np.mean(t_300)
stddev_300 = np.std(t_300)
stderr_300 = sem(t_300)


""" 70 KELVIN """
files = os.listdir(directory + "/70Kelvin")
t_70 = []
for file in files:
    T, _, _, _ = getNoisyTemp("70Kelvin/" + file, pr=False)
    t_70.append(T)

t_70 = np.array(t_70)
mean_70 = np.mean(t_70)
stddev_70 = np.std(t_70)
stderr_70 = sem(t_70)


""" 4.2KELVIN """
files = os.listdir(directory + "/4.2Kelvin")
t_42 = []
for file in files:
    T, _, _, _ = getNoisyTemp("4.2Kelvin/" + file, pr=False)
    t_42.append(T)

t_42 = np.array(t_42)
mean_42 = np.mean(t_42)
stddev_42 = np.std(t_42)
stderr_42 = sem(t_42)

""" 0.1KELVIN """
files = os.listdir(directory + "/0.1Kelvin")
t_01 = []
for file in files:
    T, _, _, _ = getNoisyTemp("0.1Kelvin/" + file, pr=False)
    t_01.append(T)

t_01 = np.array(t_01)
mean_01 = np.mean(t_01)
stddev_01 = np.std(t_01)
stderr_01 = sem(t_01)



""" Fitting Errors """
T = np.array([0.1, 4.2, 70, 300, 1000])
mean = np.array([mean_01, mean_42, mean_70, mean_300, mean_1000])
stddev = np.array([stddev_01, stddev_42, stddev_70, stddev_300, stddev_1000])
stderr = np.array([stderr_01, stderr_42, stderr_70, stderr_300, stderr_1000])


fit = np.polyfit(T[:-1], mean[:-1], deg=1, w=stderr[:-1])
poly = lambda x, fit: fit[0]*x + fit[1]


""" PLOTS """

plt.figure()
plt.errorbar(T, mean, stderr, 0, ls='', capsize = 3, label = "Mean measured $T$ Values")
plt.plot(T, poly(T, fit), ls='--', label = "Weighted Linear Fit")
plt.legend(fontsize = "x-large")
plt.xlabel("Measured Temperature (K)")
plt.ylabel("Temperature $T$ (K)")
plt.title("Mean measured Temperatures vs Real Temperature")
plt.minorticks_on()
plt.grid(b=True,which='both',color='#999999',linestyle='-',alpha=0.3)
plt.show()


plt.figure()
plt.plot(T, stderr, label = "Standard Errors", marker = 'o', ls='--')
plt.plot(T, stddev, label = "Standard Deviations", marker = 'o', ls='--')
plt.title("Standard Deviations and Errors at various Temperatures")
plt.xlabel("Temperature $T$ (K)")
plt.ylabel("Stddev and Stderr (K)")
plt.legend(fontsize = "x-large")
plt.minorticks_on()
plt.grid(b=True,which='both',color='#999999',linestyle='-',alpha=0.3)
plt.show()

""" plot errors as fractions of temperature """
plt.figure()
plt.plot(T, stderr / T, label = "Standard Errors / $T$", marker = 'o', ls='--')
plt.plot(T, stddev / T, label = "Standard Deviations / $T$", marker = 'o', ls='--')
plt.title("Fractional Standard Deviations and Errors at various $T$")
plt.xlabel("Temperature $T$ (K)")
plt.ylabel("Stddev and Stderr (K) / $T$")
plt.legend(fontsize = "x-large")
plt.minorticks_on()
plt.grid(b=True,which='both',color='#999999',linestyle='-',alpha=0.3)
plt.show()



# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
# fig.suptitle('measured temperature T with errors (K)')
# plt.minorticks_on()
# plt.grid(b=True,which='both',color='#999999',linestyle='-',alpha=0.3)
# ax1.errorbar(T[:-1], mean[:-1], stderr[:-1], 0, ls='--', color = 'b', capsize = 5)
# ax1.set(xlabel="set Temperature (K)", ylabel="measured temperature (K)")
# plt.minorticks_on()
# plt.grid(b=True,which='both',color='#999999',linestyle='-',alpha=0.3)
# ax2.errorbar(T, mean, stderr, 0, ls='--', capsize = 5)
# ax2.plot(T, poly(T, fit), label = "1")
# ax2.set(xlabel="set Temperature (K)", ylabel="measured temperature (K)")
# plt.minorticks_on()
# plt.grid(b=True,which='both',color='#999999',linestyle='-',alpha=0.3)



# plt.figure()
# plt.errorbar(T[:-1], mean[:-1], stderr[:-1], 0, ls='--', color = 'b', capsize = 5)
# plt.xlabel("set Temperature (K)", fontsize = '18')
# plt.ylabel("measured temperature (K)", fontsize = '18')
# plt.minorticks_on()
# plt.grid(b=True,which='both',color='#999999',linestyle='-',alpha=0.3)
# plt.show()

# plt.figure()
# plt.plot(T[:-1], stderr[:-1], label = "std error")
# plt.plot(T[:-1], stddev[:-1], label = "std dev")
# plt.legend()
# plt.minorticks_on()
# plt.grid(b=True,which='both',color='#999999',linestyle='-',alpha=0.3)
# plt.show()






