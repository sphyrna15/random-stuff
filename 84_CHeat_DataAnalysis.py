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

V_wat = watertxt[:,0]
Vres_wat = watertxt[:,1]
Vbr_wat = watertxt[:,2]

V_cop = coppertxt[:,0]
Vres_cop = coppertxt[:,1]
Vbr_cop = coppertxt[:,2]

V_alu = aluminiumtxt[:,0]
Vres_alu = aluminiumtxt[:,1]
Vbr_alu = aluminiumtxt[:,2]


# Scan rate "scans per second

watscan = 2.5 
copscan = 1.0
aluscan = 1.0




