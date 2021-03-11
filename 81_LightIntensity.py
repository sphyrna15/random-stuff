#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:11:29 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

#################### DATA IMPORT #############################################

# load file and return numpy array of [time, lx] data
def getData(filename):
    
    points = []
    csv = np.genfromtxt('Data/81_Data_LightIntensity/' + filename, delimiter=',', skip_header=True)
        
    return csv

filenames = [] #List of Filenames to import
filenames.append('1.1_flash1_warm-up.csv')
filenames.append('1.1_lamp1_warm-up.csv')

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

lamp_files = [] #lamp1
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

#Check distance law for lamp
lamp = getAverage(lamp_files)
lamp_model = curve_fit(fit_function, lamp[:,0], lamp[:,1], p0=[1,1.5])


flash_files = [] #flashlight files
flash_files.append(('2.01_flash1_r10cm.csv', 10))
flash_files.append(('2.02_flash1_r15cm.csv', 15))
flash_files.append(('2.03_flash1_r20cm.csv', 20))
flash_files.append(('2.04_flash1_r25cm.csv', 25))
flash_files.append(('2.05_flash1_r30cm.csv', 30))
flash_files.append(('2.06_flash1_r35cm.csv', 35))
flash_files.append(('2.07_flash1_r40cm.csv', 40))
flash_files.append(('2.08_flash1_r45cm.csv', 45))
flash_files.append(('2.09_flash1_r50cm.csv', 50))

#Check disntace law for flashlight
flash = getAverage(flash_files)
flash_model = curve_fit(fit_function, flash[:,0], flash[:,1], p0=[1,1.5])


################### DATA PLOTTING ############################################

x1 = np.linspace(10, 70, 100)
a1 = lamp_model[0][0]
p1 = lamp_model[0][1]
#lamp
plt.figure()
plt.plot(lamp[:,0], lamp[:,1], 'bx', label='Data')
plt.plot(x1, a1 / x1**p1, 'k--', label='Curve Fit')
plt.xlabel('distance $r$ from lamp (cm)')
plt.ylabel('EV (lx)')
plt.title('Lamp Distance Data')
plt.legend()
plt.grid()
plt.show()

x2 = np.linspace(10,50, 100)
a2 = flash_model[0][0]
p2 = flash_model[0][1]
#flashlight
plt.figure()
plt.plot(flash[:,0], flash[:,1], 'bx', label='Data')
plt.plot(x2, a2 / x2**p2, 'k--', label='Curve Fit')
plt.xlabel('distance $r$ from lamp (cm)')
plt.ylabel('EV (lx)')
plt.title('Flashlight Distance Data')
plt.legend()
plt.grid()
plt.show()



##################### AUFGABE 3 #############################################

data = pd.read_csv('Data/81_Data_LightIntensity/4.1_flash12_d40_l50.csv')
time = data["Time (s)"].values
lum = data["Illuminance (lx)"].values

# time = time[1:] - time[1] #hier einfach die Daten gesliced, damit der Peak in der Mitte vom array ist
# lum = lum[1:]

l = 0.50 #in m, die Distand, über die ich gemessen habe
t = time[-1] #in s, Zeitintervall in dem in gemessen habe
d = 0.40 #in m abstand zur Glüchbirne

v = l/t #in m/s

pos = time[170:]*v - time[170]*v # Zeitachse in Ortachse umgewandelt

theta = np.arctan(pos/d) #Winkel aus den Positionen berechnet
theta = np.rad2deg(theta) #in Grad umgewandelt
theta2 = -1*np.flip(theta) 

ang = np.ones(len(theta)*2)

for i in range(len(theta2)):
    ang[i] = theta2[i]

for i in range(len(theta)):
    ang[len(theta2) + i] = theta[i]
    
plt.figure()
plt.plot(ang, lum, '.', label = 'Flashlight')
plt.xlabel("Angle in deg")
plt.ylabel("light intensity in lx")
plt.legend(loc = 'best')
plt.grid()
plt.show()


