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



##################### Smallest enclosing Circle #################

x, y, radius = sec.make_circle(sp100_n3000_rs1802)

# get the fractal dimension of the cluster and plot process
def getFractDim_sec(filename, N, plot = True):
    
    radii = []
    i = 1
    while i < N:
        points = getPoints(filename, i)
        x, y, radius = sec.make_circle(points)
        radii.append((i, radius))
        i += 100
    
    if plot:
        rad = np.array(radii)
        plt.figure()
        plt.plot(np.log(rad[:,0]), np.log(rad[:,1]), label='Data')
        plt.title('LogLog fit to SEC data')
        plt.xlabel("Number of Particles")
        plt.ylabel("Smallest Enclosing Radius")
        plt.legend()
        plt.grid()
        plt.show()
    
    return rad

radii = getFractDim_sec('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000)




##################### Density Autocorrelation ###################




##################### Radius of Gyration #########################