""" Physikpraktikum 2 Experiment 72 - Fractal Clusters in Two Dimensions """
""" Task:  Given a custer generated with 72_cluster_MC.py, 
    calculate the fractal dimension of the cluser using 3 methods:
    - Smallest enclosing circle
    - Density Autocorrelation
    - Radius of Gyration                                                """

import numpy as np
import matplotlib.pyplot as plt
import smallestenclosingcircle as sec

###################### DATA IMPORT ##############################

# load file and return list of points
def getPoints(filename, maxN):
    
    points = []
    csv = np.genfromtxt('72_MonteCarlo_Clusters/' + filename, delimiter=',', skip_header=True)
    for i in range(maxN):
        points.append((csv[i,1], csv[i,2])) #append points
        
    return points

sp100_n3000_rs1802 = getPoints('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000)
sp050_n5000_rs1802 = getPoints('02_cluster_mc_rs1802_n5000_p0.50.csv', 5000)


##################### Smallest enclosing Circle #################

x, y, radius = sec.make_circle(sp050_n5000_rs1802)

# get the fractal dimension of the cluster and plot process
def getFractDim_sec(filename, N, plot = True):
    
    radii = []
    i = 1
    while i < N:
        points = getPoints(filename, i)
        x, y, radius = sec.make_circle(points)
        radii.append((i, radius))
        i += 100
        
    #linear fit to get fractal dimension
    rad = np.array(radii)
    rad = rad[1:, :]
    x = np.log(rad[:,0]) ; y = np.log(rad[:,1])
    model = np.polyfit(x, y, 1)
    
    if plot:
        plt.figure()
        #plt.plot(rad[:,0], rad[:,1])
        plt.plot(np.log(rad[:,0]), np.log(rad[:,1]), label='Data')
        #plt.loglog(rad[:,0], rad[:,1], label='Data')
        plt.plot(x, model[0]*x + model[1], label='Log Fit')
        plt.title('LogLog fit to SEC data')
        plt.xlabel("Number of Particles")
        plt.ylabel("Smallest Enclosing Radius")
        plt.legend()
        plt.grid()
        plt.show()
    
    return rad, model

radii, model = getFractDim_sec('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000)
rad2, mod2 = getFractDim_sec('02_cluster_mc_rs1802_n5000_p0.50.csv', 5000)



##################### Density Autocorrelation ###################




##################### Radius of Gyration #########################