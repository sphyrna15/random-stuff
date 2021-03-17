""" Physikpraktikum 2 Experiment 72 - Fractal Clusters in Two Dimensions """
""" Task:  Given a custer generated with 72_cluster_MC.py, 
    calculate the fractal dimension of the cluser using 3 methods:
    - Smallest enclosing circle
    - Density Autocorrelation
    - Radius of Gyration                                                """

import math
import numpy as np
import matplotlib.pyplot as plt
import smallestenclosingcircle as sec

###################### DATA IMPORT ##############################

# load file and return list of points
def getPoints(filename, maxN):
    
    points = []
    csv = np.genfromtxt('Data/72_MonteCarlo_Clusters/' + filename, delimiter=',', skip_header=True)
    for i in range(maxN):
        points.append((csv[i,1], csv[i,2])) #append points
        
    return points

# Example
sp100_n3000_rs1802 = getPoints('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000)
# sp050_n5000_rs1802 = getPoints('02_cluster_mc_rs1802_n5000_p0.50.csv', 5000)

# make a list of all the clusters to easily collect results
cluster_list = []
# cluster_list.append(('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000))
# cluster_list.append(('02_cluster_mc_rs1802_n5000_p0.50.csv', 5000))
""" clusters for analysing effect of sticking probability """
cluster_list.append(('03_cluster_mc_rs1701_n15000_p1.00.csv', 15000))
cluster_list.append(('04_cluster_mc_rs1901_n5000_p1.00.csv', 5000))
cluster_list.append(('05_cluster_mc_rs1901_n12500_p0.25.csv', 12500))
cluster_list.append(('06_cluster_mc_rs1501_n7500_p0.25.csv', 7500))
cluster_list.append(('07_cluster_mc_rs1807_n10000_p0.10.csv', 10000))
cluster_list.append(('08_cluster_mc_rs1897_n8000_p0.10.csv', 8000))
cluster_list.append(('09_cluster_mc_rs1597_n12000_p0.05.csv', 12000))
cluster_list.append(('10_cluster_mc_rs1997_n6000_p0.05.csv', 6000))
cluster_list.append(('11_cluster_mc_rs1757_n9000_p0.01.csv', 9000))
cluster_list.append(('12_cluster_mc_rs1802_n6000_p0.01.csv', 6000))
""" clusters varying random seed """
cluster_list.append(('13_cluster_mc_rs1753_n9000_p0.50.csv', 9000))
cluster_list.append(('14_cluster_mc_rs1973_n9000_p0.50.csv', 9000))
cluster_list.append(('15_cluster_mc_rs1815_n9000_p0.50.csv', 9000))
cluster_list.append(('16_cluster_mc_rs1517_n9000_p0.50.csv', 9000))
cluster_list.append(('17_cluster_mc_rs1901_n9000_p0.50.csv', 9000))

##################### Smallest enclosing Circle (SEC) #################


# get the fractal dimension of the cluster and plot process
def sec_radius(filename, N, plot = True):
    # Filename of Cluster
    # N : number of particles in cluster
    # plot : boolean - do you want to plot the results?
    
    allPoints = getPoints(filename, N)
    radii = []
    for i in range(100, N, 100):
        points = allPoints[:i]
        x, y, radius = sec.make_circle(points)
        radii.append((i, radius))
        
        
    #linear fit to get fractal dimension
    rad = np.array(radii)
    rad = rad[1:, :]
    y = np.log(rad[:,0]) ; x = np.log(rad[:,1]) # x number of particles
    model = np.polyfit(x, y, 1, full=True)
    # see manual equation (13):
    fractDim = model[0]
    
    if plot:
        plt.figure()
        plt.plot(x, y, 'r', label='Data')
        plt.plot(x, model[0]*x + model[1], 'k--', label='Poly Fit')
        plt.title('LogLog fit to SEC data')
        plt.xlabel("Number of Particles")
        plt.ylabel("Smallest Enclosing Radius")
        plt.legend()
        plt.grid()
        plt.show()
    
    #returns radii and model parameters [fractal dimension, constant]
    return fractDim, rad, model


##################### Density Autocorrelation ###################

def density_correlation(filename, Nmax, delta_r = 1, plot = True):
    
    allPoints = getPoints(filename, Nmax)
    N = len(allPoints)
    correlations = []
    points = np.array(allPoints)
    
    c = 0.0    
    _, _, radius = sec.make_circle(allPoints)
    R = np.linspace(radius/4, radius/2, 15)
    
    for r in R:
        print(r)
        for i in range(N):
             distances = np.linalg.norm(points - points[i], axis=1)
             check = distances - r
             inst = np.logical_and(- delta_r / 2 < check, check < delta_r / 2)
             c += np.count_nonzero(inst)
        C = c / (N*4*np.pi*r*delta_r)
        correlations.append((C, r))
            
    
    # Linear Fit    
    results = np.array(correlations) 
    x = np.log(results[:,1]) ; y = np.log(results[:,0])
    model = np.polyfit(x, y, 1)
    fractDim = model[0] + 2 #Fractal Dimension
    #print(model[0])
    if plot:
        plt.figure()
        plt.plot(x, y, label='Data')
        plt.plot(x, model[0]*x + model[1], label='Poly Fit')
        plt.title('LogLog fit to DAC data')
        plt.xlabel("Length of Radius")
        plt.ylabel("Value of C(r)")
        plt.legend() 
        plt.grid()
        plt.show()
    
    return fractDim, correlations, model
    
    
##################### Radius of Gyration #########################

def radius_of_gyration(filename, N, plot = True):
    # Filename of Cluster
    # N : number of particles in cluster
    # plot : boolean - do you want to plot the results?
    
    allPoints = getPoints(filename, N)
    gyrations = [] # Radii of gyration
    for i in range(100, N, 100):
        points = allPoints[:i] #get points
        points = np.array(points) #make them numpy compatible
        # drehpunkt berechnen
        drehpunkt = np.mean(points, axis = 0)
        # radius of gyration - see manual equation (10)
        radsum = np.sum((points - drehpunkt)**2)
        gyration = np.sqrt(radsum / i) # Radius of gyration
        gyrations.append((i, gyration)) # add all radii
        

    # linear fit to get fractal dimension
    gyr = np.array(gyrations)
    x = np.log(gyr[:,0]) ; y = np.log(gyr[:,1])  # x number of particles
    model = np.polyfit(x, y, 1, full=True)
    # see manual equation (11)
    beta = model[0]
    fractDim = 1 / beta
    
    # Plot results if wanted
    if plot:
        plt.figure()
        plt.plot(x, y, 'r', label='Data')
        plt.plot(x, model[0]*x + model[1], 'k--', label='Poly Fit')
        plt.title('LogLog fit to ROG data')
        plt.xlabel("Number of Particles")
        plt.ylabel("Radius of Gyration")
        plt.legend()
        plt.grid()
        plt.show()
    
    return fractDim, gyrations, model
    
    
if __name__ == '__main__':
    
    # print('Smallest Enclosing Circle')
    # #Smallest enclosing Circle:
    # sec_results = [("dataset", "fractal dimension")] #results with SEC
    # sec_residuals = []
    # for (filename, N) in cluster_list:
    #     D, _, model = sec_radius(filename, N, plot=False)
    #     print(filename+ ' done')
    #     sec_residuals.append(model[1])
    #     sec_results.append((filename, D))
    
    print('---' * 20)
    print('Density Auto Correlation')
    #Density Auto Correlation
    dac_results = [("dataset", "fractal dimension")] #results with radius of gyration
    for (filename, N) in cluster_list:
        D, _, _ = density_correlation(filename, N, plot=True)
        print(filename +' done ' + str(D))
        dac_results.append((filename, D))       
    
    # print('---' * 20)
    # print('Radius of Gyration')
    # #Radius of Gyration
    # rog_results = [("dataset", "fractal dimension")] #results with radius of gyration
    # rog_residuals =[]
    # for (filename, N) in cluster_list:
    #     D, _, model = radius_of_gyration(filename, N, plot=False)
    #     print(filename + ' done')
    #     rog_residuals.append(model[1])
    #     rog_results.append((filename, D))
        
        
        
        
    

