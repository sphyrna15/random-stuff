#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:11:29 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#################### DATA IMPORT #############################################

# load file and return numpy array of [time, lx] data
def getData(filename):
    
    points = []
    csv = np.genfromtxt('Data/81_Data_LightIntensity/' + filename, delimiter=',', skip_header=True)
        
    return csv

filenames = [] #List of Filenames to import
filenames.append('1.1_flash1_warm-up.csv')
filenames.append('1.1_lamp1_warm-up.csv')
filenames.append('2.01_flash1_r10cm.csv')
filenames.append('2.01_lamp1_r10cm.csv')
filenames.append('2.02_lamp1_r15cm.csv')
filenames.append('2.03_lamp1_r20cm.csv')
filenames.append('2.04_lamp1_r25cm.csv')
filenames.append('2.05_lamp1_r30cm.csv')
filenames.append('2.06_lamp1_r35cm.csv')
filenames.append('2.07_lamp1_r40cm.csv')
filenames.append('2.08_lamp1_r45cm.csv')
filenames.append('2.02_lamp1_r15cm.csv')

example = getData(filenames[0])


# get 20s average of a list of files and put them in 1 numpy array
def getAverage(files): #files list (filename, radius)
    
    points = []
    for file, r in files:
        #only use 20s
        data = getData(file)
        idx = np.where(data[:,0] <= 20)[0]
        data = data[:idx[-1], :]
        avg = np.average(data)
        points.append((r, avg))
    
    return np.array(points)

fit_function = lambda r, a, x : a / (r**x) #curve for fitting

lamp_files = []
lamp_files.append(('2.01_lamp1_r10cm.csv', 10))
lamp_files.append(('2.02_lamp1_r15cm.csv', 15))
lamp_files.append(('2.03_lamp1_r20cm.csv', 20))
lamp_files.append(('2.04_lamp1_r25cm.csv', 25))
lamp_files.append(('2.05_lamp1_r30cm.csv', 30))
lamp_files.append(('2.06_lamp1_r35cm.csv', 35))
lamp_files.append(('2.07_lamp1_r40cm.csv', 40))
lamp_files.append(('2.08_lamp1_r45cm.csv', 45))
lamp_files.append(('2.09_lamp1_r50cm.csv', 50))
lamp_files.append(('2.10_lamp1_r55cm.csv', 55))
lamp_files.append(('2.11_lamp1_r60cm.csv', 60))        
lamp_files.append(('2.12_lamp1_r65cm.csv', 65))
lamp_files.append(('2.13_lamp1_r70cm.csv', 70))

lamp = getAverage(lamp_files)
lamp_model = curve_fit(f=fit_function, xdata=lamp[:,0], ydata=lamp[:,1], p0=[1,1.5])



################### DATA PLOTTING ############################################

x = np.linspace(10, 70, 100)
a = lamp_model[0][0]
p = lamp_model[0][1]
#lamp
plt.figure()
plt.plot(lamp[:,0], lamp[:,1], 'bx', label='Data')
plt.plot(x, a / x**p, 'k--', label='Curve Fit')
plt.xlabel('distance $r$ from lamp (cm)')
plt.ylabel('EV (lx)')
plt.title('Lamp Distance Data')
plt.legend()
plt.grid()
plt.show()

