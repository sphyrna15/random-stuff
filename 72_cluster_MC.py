# written by William Huxter (2020)
# inspired by the work of Witten and Sander who started the field of diffusion limited aggregation
# this code is meant to reproduce their original work (with a square grid)
# see the following papers for more details
# T. A. Witten and L. M. Sander, Phys. Rev. Lett. 47 (1981).
# https://doi.org/10.1103/PhysRevLett.47.1400
# P. Meakin, Phy. Rev. A 27 (1983).
# https://doi.org/10.1103/PhysRevA.27.1495

import math
import matplotlib.pyplot as plt
import random
import smallestenclosingcircle
import time


def distance(u,v):
    return math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)


def place_on_cirlce(particle, circ_x, circ_y, radius):
    # Uses Green's function for diffusion equation on a disc.
    # Places a particle (at [x0, y0] outside of the disc) on a disc 
    # of radius Rc via a probabilistic approach.
    # L. M. Sander, Contemporary Physics, 41 (2000).  page 213 for details
    # https://doi.org/10.1080/001075100409698
    xi = random.random()  # random number form 0 to 1
    x0 = particle[0] - circ_x
    y0 = particle[1] - circ_y
    r0 = math.sqrt(x0 ** 2 + y0 ** 2)
    Rc = radius
    V = ((r0 - Rc) / (r0 + Rc)) * math.tan(math.pi * xi)
    X = (Rc / r0) * ((1 - V ** 2) * x0 - 2 * V * y0) / (1 + V ** 2)
    Y = (Rc / r0) * ((1 - V ** 2) * y0 + 2 * V * x0) / (1 + V ** 2)
    return [int(math.ceil(X + circ_x)),  int(math.ceil(Y + circ_y))]


def update_eff_radii(radius, particle_radius):
    # returns the radius at which particles are placed
    eff_r_check = int(math.ceil((radius + 5*particle_radius)))
    # used for starting positions and plotting limits 
    eff_rl = int(math.ceil((radius + 15*particle_radius)))
    # used as maximum distance to move particle away
    return eff_r_check, eff_rl


def get_quadrant(pos):
    # helper function used to speed up checking many positions and distances
    if pos[0] > 0 and pos[1] >= 0:
        return 0
    elif pos[0] <= 0 and pos[1] > 0:
        return 1
    elif pos[0] < 0 and pos[1] <= 0:
        return 2
    else:  # pos[0] >= 0 and pos[1] < 0 :
        return 3


def add_to_stickps(stickps, stickdir, bin_size, particle=[0, 0]):
    # subfunction used when adding a particle to the cluster
    # must track available point to stick too and add them in the correct place
    for dir in stickdir:
        # first step is to add to the correct sublist
        new_sp = [particle[0] + dir[0], particle[1] + dir[1]]
        r = distance(new_sp, [0, 0]) / (0.5 * bin_size)
        index = int(math.ceil(r)) - 1
        quad = get_quadrant(new_sp)
        if new_sp not in stickps[index][quad]:
            # only add if it is not already there
            stickps[index][quad].append(new_sp)


def add_to_cluster(stickps, particle, cluster_parts, cluster, bin_size):
    # subfunction ued when adding a particle to the cluster
    # similar to 'add_to_stickps'
    r = distance(particle, [0, 0]) / (0.5 * bin_size)
    index = int(math.ceil(r)) - 1
    quad = get_quadrant(particle)
    cluster_parts[index][quad].append(particle)
    stickps[index][quad].remove(particle)
    cluster.append(particle)


def build_intersection(position, parts, bin_size):
    # subfunction used when random walking
    # builds a set of points to check for intersections
    r = distance(position, [0, 0]) / (0.5 * bin_size)
    index = int(math.ceil(r)) - 1
    quad = get_quadrant(position)
    
    # takes +/- 1 indexes to make sure edge cases are caught
    intersections = parts[index][quad] + parts[index - 1][quad] + parts[index + 1][quad]
    if position[0] == 0:  # along x-axis, need to add another quadrant to be safe
        if quad == 0:
            quad_pair = 3
        else:  # quad == 2:
            quad_pair = 1
        intersections += parts[index][quad_pair] + parts[index - 1][quad_pair] + parts[index + 1][quad_pair]

    if position[1] == 0:  # along y-axis, need to add another quadrant to be safe
        if quad == 1:
            quad_pair = 0
        else:  # quad == 3:
            quad_pair = 2
        intersections += parts[index][quad_pair] + parts[index - 1][quad_pair] + parts[index + 1][quad_pair]
    return intersections


def cluster_mc(sp=1.00, Nmax=1000, rs=0):
    # main function that runs the monte carlo code used for creating the clusters
    # sp = sticking probability, = 1 is the classical DLA case, 0 < sp =< 1 are the bounds
    # Nmax = maximum number of particles in the cluster, the functions runs until Nmax is hit
    # rs = random seed (just a number) that set the pseudo-random number generator
    # the random seed is what makes this a Monte Carlo simulation
    
    # some initial checks
    if not 0 < sp <= 1:
        print('ERROR: bounds for sticking probability (sp) are incorrect (must have 0 < sp <= 1). Stopping the simulation.')
        return
    if type(Nmax) is not int:
        print('ERROR: maximum particle number (Nmax) must be type int. Stopping the simulation.')
        return
    rs = int(rs)  # for file saving it is more convenient to have an integer
    random.seed(rs)  # set the random seed
    
    # file opening
    strname = 'cluster_mc_rs%d_n%d_p%0.2f'%(rs, Nmax, sp)
    fh = open(strname + '.csv','w')
    fh.write('number,x,y\n')
    fh.write('1,0,0\n')

    walkdir = [[1,0],[0,1],[-1,0],[0,-1]]       # particles only walk in 4 directions
    seed = [0,0]                                # single frozen seed
    cluster = [seed]                            # just a list of points
    clustsize = len(cluster)

    stickps = []                                # 3D array of sticking points
    cluster_parts = []                          # 3D array of particles in the cluster
    bin_size = 3                                # hard coded value to shape the 3D arrays
    stickdir = [[1,0],[0,1],[-1,0],[0,-1]]      # particle can attach only at 4 sites relative to other particles
    
    for i in range(int(Nmax/bin_size) +10):     # approximation to scale subintervals based on Nmax
        stickps.append([[],[],[],[]])           # 4 lists for the 4 quadrants of the grid
        cluster_parts.append([[],[],[],[]])     # 4 lists for the 4 quadrants of the grid

    # initialize the sublists
    add_to_stickps(stickps, stickdir, bin_size)
    cluster_parts[0][0].append(cluster[0])
    cluster_parts[0][1].append(cluster[0])
    cluster_parts[0][2].append(cluster[0])        
    cluster_parts[0][3].append(cluster[0])

    # circle is a list of three values --> [x_center, y_center, radius]
    circ = smallestenclosingcircle.make_circle(cluster)
    eff_r_check, eff_rl = update_eff_radii(circ[2], 0.5)

    # set some controlling flag
    active_particle = False  # must initialize a random molecule
    rand_walk = False
    
    # now  we can enter the main loop
    while clustsize < Nmax:
        if not active_particle:
            # generate a random particle along the edge
            theta = random.uniform(0,2*math.pi)
            start_x = int(math.ceil(circ[0] + eff_r_check*math.cos(theta)))
            start_y = int(math.ceil(circ[1] + eff_r_check*math.sin(theta)))
            particle = [start_x, start_y]

            # skip random walk step and set active_particle flag
            active_particle = True
            rand_walk = False
            
        # random walk of active particle
        if rand_walk:
            # time to move to a new position, by randomly picking a walk direction
            delta = random.randint(0,3)
            temp_pos = [particle[0] + walkdir[delta][0], particle[1] + walkdir[delta][1]]
            intersectpossibility = build_intersection(temp_pos, cluster_parts, bin_size)

            while temp_pos in intersectpossibility: # keep particle out of cluster (needed when sp < 1)
                delta = random.randint(0,3)
                temp_pos = [particle[0] + walkdir[delta][0], particle[1] + walkdir[delta][1]]
                intersectpossibility = build_intersection(temp_pos, cluster_parts, bin_size)

            if distance(temp_pos, [circ[0], circ[1]]) >= eff_rl:
                # reset particle if too far away fromthe circle
                temp_pos = place_on_cirlce(temp_pos, circ[0], circ[1], eff_r_check)
            
            # if eveything checks out, we can update the position of the particle
            particle = temp_pos

        # check to see if the particle should be added to the cluster
        if distance(particle,[circ[0],circ[1]]) <= eff_r_check: # no need to try if the particle is outside the outside circle

            stickps_intersection = build_intersection(particle, stickps, bin_size)
            
            # if the particle is in a position to attach to the cluster
            if particle in stickps_intersection:
                prob = random.random()  # number between 0 and 1
                if prob < sp:
                    # the particle should be added to the cluster
                    add_to_stickps(stickps, stickdir, bin_size, particle)
                    add_to_cluster(stickps, particle, cluster_parts, cluster, bin_size)
                    circ = smallestenclosingcircle.make_circle(cluster)
                    eff_r_check, eff_rl = update_eff_radii(circ[2], 0.5)
                    # set the flag to make a new a particle
                    active_particle = False
            else:
                rand_walk = True
        else:
            rand_walk = True
            
        # if you have a new particle, add it to the data
        if len(cluster) > clustsize:
            clustsize = len(cluster)
            
            fh.write('%d,%d,%d\n'%(len(cluster),particle[0],particle[1]))
            
            if clustsize%500 == 0:
                # print an update every n particles (as a sanity check)
                print('particle', len(cluster), 'at position:', particle)
            
            if clustsize == Nmax:
                # close the data file and make the picture of the cluster
                fh.close()
                
                fig, ax = plt.subplots(figsize=(10,10))
                for particle in cluster:
                    plt.scatter(particle[0],particle[1], color='k',s=10)
                ax.set_xlim(circ[0]-eff_r_check,circ[0]+eff_r_check)
                ax.set_ylim(circ[1]-eff_r_check,circ[1]+eff_r_check)
                ax.axis('off')
                fig.savefig(fname=strname + '.png', dpi=600, bbox_inches='tight')
                

if __name__ == "__main__":
    t0 = time.time()
    cluster_mc(sp=0.5, Nmax=9000, rs=1901)
    t1 = time.time()
    print('Total time (in seconds):', t1-t0)

