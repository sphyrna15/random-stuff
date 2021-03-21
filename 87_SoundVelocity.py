#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:57:13 2021

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import solve, norm
from scipy.linalg import qr
import scipy as sp
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from lmfit import Model
from uncertainties import ufloat
from uncertainties import unumpy


######################## DATA IMPORT ########################################


# load file and return numpy array of [time, lx] data
def getData(filename):
    
    points = []
    csv = np.genfromtxt('Data/87_Data_SoundVelocity/' + filename, delimiter=',', skip_header=True)
        
    return csv

filenames = []
#background noise
filenames.append('1_background-01.csv')
filenames.append('1_background-02.csv')
filenames.append('1_background-03.csv')
# both ends open
filenames.append('1_2open-01.csv')
filenames.append('1_2open-02.csv')
filenames.append('1_2open-03.csv')
# filenames.append('1_2open-04.csv') #did not work
# one end open
# filenames.append('1_1open-01.csv') #did not work
filenames.append('1_1open-02.csv')
filenames.append('1_1open-03.csv')
filenames.append('1_1open-04.csv')
filenames.append('1_1open-05.csv')


########################## TASK 1 ###########################################

def Lorentz(x , y0, a , xc , w):
    return y0 + (2*a / np.pi) * (w / (4 * (x-xc)**2 + w**2))


def task1(filename, plot = False, plot2=False, f0=400):
    data = pd.read_csv('Data/87_Data_SoundVelocity/' + filename)
    amp = data["Recording (a.u.)"].values
    time = data["Time (ms)"].values
    
    if plot2:
        plt.figure()
        plt.plot(time, amp)
        plt.xlabel("time in ms")
        plt.ylabel("Amplitude")
        plt.title("Raw Data " + filename)
        plt.show()
    
    freq_fft = np.fft.fftfreq(len(time) ,(time[2] - time[1])/1000)
    amp_fft = np.abs(np.fft.fft(amp))
    
    data_length =int(len(time) *0.5)
    freq_fft_plot = freq_fft[:data_length]
    amp_fft_plot = amp_fft[:data_length]
    
    # Fitting 
    lmodel = Model(Lorentz)
    result = lmodel.fit(amp_fft_plot, x=freq_fft_plot, y0=0.8, a=100, xc=f0, w=20)
    
    conf = result.eval_uncertainty(sigma =1)
    print(result.fit_report())
    
    f = result.best_values.get('xc')
    y = result.best_values.get('y0')
    a = result.best_values.get('a')
    w = result.best_values.get('w')
    f0err = np.sqrt(result.covar[2 ,2])
    
    # # Other Fit Method
    # result = curve_fit(Lorentz, freq_fft_plot, amp_fft_plot, p0=[0.8, 100, 500, 20, 1, 1])
    # y = result[0][0]
    # a = result[0][1]
    # f = result[0][2]
    # w = result[0][3]
    # b = result[0][4]
    # b0 = result[0][5]
    
    if plot:
        plt.figure()
        plt.plot(freq_fft_plot[:1500], amp_fft_plot[:1500], label = "FFT")
        plt.plot(freq_fft_plot[:1500], Lorentz(freq_fft_plot, y, a, f, w)[:1500], 'r', label = "fit")
        plt.grid()
        plt.legend()
        plt.xlabel("frequency f in Hz")
        plt.ylabel("Amplitude")
        plt.title("FFT of " + filename)
        plt.show()
        
    return result, (freq_fft, amp_fft)

############## TASK 1a - both ends open ####################################
files1a = []
files1a.append('1_2open-01.csv')
files1a.append('1_2open-02.csv')
files1a.append('1_2open-03.csv')
# files1a.append('1_2open-04.csv')

# results1a = []
# for file in files1a:
#     model, data = task1(file, plot=True)
#     results1a.append(model)

model1a, data = task1(files1a[0], plot=True, f0=400)

################### TASK 1b - one end closed ###############################
files1b = []
# files1b.append('1_1open-01.csv')
files1b.append('1_1open-02.csv')
files1b.append('1_1open-03.csv')
files1b.append('1_1open-04.csv')
files1b.append('1_1open-05.csv')
# files1b.append('1_1open-06.csv') #hand

# results1b = []
# for file in files1b:
#     model, data = task1(file, plot=True)
#     results1b.append(model)
    
model1b, data = task1(files1b[2], plot=True, f0=200)



####################### TASK 2 ############################################

#Multi Lorentian
def M(x, b0, b, y0, a0, a1, a2, a3, f0, f1, f2,  f3, w0, w1, w2, w3):
    return (y0 + b0*np.exp(1/b) + 
            (2*a0/np.pi)*(w0/(4*(x - f0)**2 + w0 **2)) + 
            (2*a1/np.pi)*(w1/(4*(x - f1)**2 + w1**2)) + 
            (2.*a2/np.pi)*(w2/(4*(x - f2)**2 + w2**2)) +
            (2.*a3/np.pi)*(w3/(4*(x - f3)**2 + w3**2)))

# give starting values for fraquencies f = [f0, f1, f2, f3]
def task2(filename, plot = False, f = [400, 800, 1200, 1600]):
    data = pd.read_csv('Data/87_Data_SoundVelocity/' + filename)
    amp = data["Recording (a.u.)"].values
    time = data["Time (ms)"].values
    
    freq_fft = np.fft.fftfreq(len(time) ,(time[2] - time[1])/1000)
    amp_fft = np.abs(np.fft.fft(amp))
    
    data_length =int(len(time) *0.5)
    freq_fft_plot = freq_fft[:data_length]
    amp_fft_plot = amp_fft[:data_length]
    
    #fitting
    lmodel3 = Model(M)
    result3 = lmodel3.fit(amp_fft_plot, x = freq_fft_plot, b0=10, b=1, y0=0.8, 
                          a0=100, a1=70, a2=50, a3=30, 
                          f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], 
                          w0 =20, w1 = 10, w2 = 5, w3 = 5.)
    
    conf = result3.eval_uncertainty(sigma =1)
    print(result3.fit_report())
    
    f0 = float(result3.best_values.get('f0'))
    f1 = float(result3.best_values.get('f1'))
    f2 = float(result3.best_values.get('f2'))
    f3 = float(result3.best_values.get('f3'))
    b =  float(result3.best_values.get('b'))
    b0 = float(result3.best_values.get('b0'))
    y0 = float(result3.best_values.get('y0'))
    a0 = float(result3.best_values.get('a0'))
    a1 = float(result3.best_values.get('a1'))
    a2 = float(result3.best_values.get('a2'))
    a3 = float(result3.best_values.get('a3'))
    w0 = float(result3.best_values.get('w0'))
    w1 = float(result3.best_values.get('w1'))
    w2 = float(result3.best_values.get('w2'))
    w3 = float(result3.best_values.get('w3'))
    f0err3 = np.sqrt(result3.covar[2 ,2])
    
    if plot:
        plt.figure()
        plt.plot(freq_fft_plot[:1200], amp_fft_plot[:1200], label = "FFT")
        #plt.plot(freq_fft_plot[:1500], Lorentz(freq_fft_plot, y, a, f, w)[:1500],  label = "fit_lorentz")
        plt.plot(freq_fft_plot[:1200], 
                 M(freq_fft_plot, b0, b, y0, a0, a1, a2, a3, f0, f1, f2, f3, w0, w1, w2, w3)[:1200], 
                 'r', label = "fit")
        plt.grid()
        plt.legend(loc ='best')
        plt.xlabel("frequency $f$ (Hz)")
        plt.ylabel("Amlitude")
        plt.title("Multilorentian Fit to " + filename)
        plt.show()
        
    
    return result3

# results2a = []
# for file in files1a:
#     model = task2(file, plot=True, f=[400, 750, 1100, 1500])
#     results2a.append(model)

model2a = task2(files1a[0], plot=True, f=[400, 750, 1100, 1500])

# results2b = []
# for file in files1b:
#     model = task2(file, plot=True, f=[200, 400, 600, 800])
#     results2b.append(model)

model2a = task2(files1b[2], plot=True, f=[200, 400, 600, 800]) 

#linear fit and error calculation - 
# input calculated resonance frequencies with respective orders + errors
def task2c(filename, freq, order, err, plot=False):
    
    freq = unumpy.uarray(freq, err)
    order = np.array(order)
    
    freq_nom = unumpy.nominal_values(freq)
    freq_err = unumpy.std_devs(freq)
    
    p1= np.polyfit(order,freq_nom,1)
    p1_err = np.polyfit(order,freq_err,1)
    print("Slope")
    print(p1[0])
    print("Error")
    print(p1_err[0])
    
    if plot:
        plt.figure()
        plt.errorbar(order,freq_nom,freq_err , 0, '.', color = 'red', label="resonance frequencies")
        plt.plot(order, freq_nom, 'ro')
        plt.plot(order, np.polyval(p1, order), 'k--' ,label="linear fit")
        plt.xlabel('order')
        plt.ylabel('resonance frequency in Hz')
        plt.title("Resonance Frequencies Linear Fit " + filename)
        plt.grid(b=True,color='#999999',linestyle='-',alpha=0.3)
        plt.xticks(np.arange(1, 3, step=1))
        plt.legend(loc = 'best')
        plt.show()
    
    return p1[0], p1_err[0] #returns slope and error


m1, m1_err = task2c(files1a[0], [388.9, 779.5, 1180.5], [1, 2, 3], [0.2, 0.7, 1.4], plot=True)
m2, m2_err = task2c(files1a[0], [258.9, 616.5], [1, 3], [0.8, 1.4], plot=True)



L = ufloat(0.43, 0.002)

m1 = ufloat(395.8, 0.6)
m2 = ufloat(178.8, 0.3)
v1 = 2*m1*L 
v2 = 4*m2*L

print(v1)
print(v2)

    
    


