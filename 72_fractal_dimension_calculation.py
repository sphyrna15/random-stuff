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

# Example
sp100_n3000_rs1802 = getPoints('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000)
# sp050_n5000_rs1802 = getPoints('02_cluster_mc_rs1802_n5000_p0.50.csv', 5000)

# make a list of all the clusters to easily collect results
cluster_list = []
cluster_list.append(('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000))
cluster_list.append(('02_cluster_mc_rs1802_n5000_p0.50.csv', 5000))


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
    model = np.polyfit(x, y, 1)
    # see manual equation (13):
    fractDim = model[0]
    
    if plot:
        plt.figure()
        plt.plot(x, y, 'r', label='Data')
        plt.plot(x, model[0]*x + model[1], 'k--', label='Log Fit')
        plt.title('LogLog fit to SEC data')
        plt.xlabel("Number of Particles")
        plt.ylabel("Smallest Enclosing Radius")
        plt.legend()
        plt.grid()
        plt.show()
    
    #returns radii and model parameters [fractal dimension, constant]
    return fractDim, rad, model

# D1, rad1, mod1 = sec_radius('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000)
# D2, rad2, mod2 = sec_radius('02_cluster_mc_rs1802_n5000_p0.50.csv', 5000)

sec_results = [("dataset", "fractal dimension")] #results with SEC
for (filename, N) in cluster_list:
    D, _, _ = sec_radius(filename, N, plot=False)
    sec_results.append((filename, D))


##################### Density Autocorrelation ###################




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
    model = np.polyfit(x, y, 1)
    # see manual equation (11)
    beta = model[0]
    fractDim = 1 / beta
    
    # Plot results if wanted
    if plot:
        plt.figure()
        plt.plot(x, y, 'r', label='Data')
        plt.plot(x, model[0]*x + model[1], 'k--', label='Log Fit')
        plt.title('LogLog fit to SEC data')
        plt.xlabel("Number of Particles")
        plt.ylabel("Smallest Enclosing Radius")
        plt.legend()
        plt.grid()
        plt.show()
    
    return fractDim, gyrations, model

# D1, gyr1, mo1 = radius_of_gyration('01_cluster_mc_rs1802_n3000_p1.00.csv', 3000)
# D2, gyr2, mo2 = radius_of_gyration('02_cluster_mc_rs1802_n5000_p0.50.csv', 5000)

rog_results = [("dataset", "fractal dimension")] #results with radius of gyration
for (filename, N) in cluster_list:
    D, _, _ = radius_of_gyration(filename, N, plot=False)
    rog_results.append((filename, D))
    
    
        
        
        
        
        
        
    

