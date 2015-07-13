#!/usr/bin/env python
"""
Library to perform DPD simulations

pv278@cam.ac.uk, 02/07/15
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import sqrt

##### global constants
kB = 1
gamma = 1

class Particle:
    """Contains properties of one particle: 
       position, velocity & type"""
    def __init__(self, pos=np.array([0, 0, 0]), vel=np.array([0, 0, 0]), typ="A"):
        self.pos = pos
        self.vel = vel
        self.typ = typ
        
    def random_init(self, L=10, T=300, typ="A"):
        self.pos = np.random.rand(3)*L
        self.vel = np.random.randn(3)*kB*T # m = 1
        self.typ = typ
    
    def __repr__(self):
        return str(self.pos)
    
         
class System(Particle):
    """
    A class to treat a system of particles, 
    inheriting from Particle including metadata: 
    -- box size
    -- number
    -- temperature
    """
    def __init__(self, N=10, L=10, T=300):
        self.L = L
        self.N = N
        self.T = T
        self.sys = []
        for i in range(self.N):
            self.sys.append(Particle())
            
    def random_init(self, seed=0):
        """Initialise with random velocities"""
        np.random.seed(seed)
        for i in range(self.N):
            self.sys[i].random_init(self.L, self.T)
            
    def __getitem__(self, i):
        return self.sys[i]
        
    def __repr__(self):
        s = ""
        for i in range(self.N):
            s += str(self.sys[i]) + "\n"
        return s
        
    def pos(self, i):
        return self.sys[i].pos
        
    def vel(self, i):
        return self.sys[i].vel
        
    def pos_table(self):
        """Table of all positions"""
        p = []
        for i in range(self.N):
            p.append(self.sys[i].pos)
        return np.array(p)
        
    def vel_table(self):
        """Table of all velocities"""
        v = []
        for i in range(self.N):
            v.append(self.sys[i].vel)
        return np.array(v)
        
    def energy(self):
        E = 0
        for i in range(self.N):
            E += LA.norm(self[i].vel)**2/2
        return E
        
    def lookup_table(self):
        """List of pairs of particles in each others' spheres
           of interaction, accounting for PBCs with flags:
           -1: direct
            0: across x-dir wall
            1: across y-dir wall
            2: across z-dir wall
        """
        table = []
        for i in range(self.N):
            for j in range(i):
                r_d = LA.norm(self.sys[i].pos - self.sys[j].pos)
                r_x = LA.norm(border_dist(self.sys[i].pos, self.sys[j].pos, 0, self.L))
                r_y = LA.norm(border_dist(self.sys[i].pos, self.sys[j].pos, 1, self.L))
                r_z = LA.norm(border_dist(self.sys[i].pos, self.sys[j].pos, 2, self.L))
                #table.append((i, j, r_d, r_x, r_y, r_z))  # testing
                if r_d < 1:
                   table.append((i, j, -1, r_d))
                elif r_x < 1:
                   table.append((i, j, 0, r_x))
                elif r_y < 1:
                   table.append((i, j, 1, r_y))
                elif r_z < 1:
                   table.append((i, j, 2, r_z))
        return table

    def force_table(self):
        """table of forces on the particles, based on lookup table"""
        ltable = self.lookup_table()
        Ftable = np.zeros((self.N, 3))
        for pair in ltable:
            if pair[2] == -1:   # direct
                r = self.sys[pair[0]].pos - self.sys[pair[1]].pos
                v = self.sys[pair[0]].vel - self.sys[pair[1]].vel
                Ftable[pair[0]] += F_tot(r, v, self.T)
                Ftable[pair[1]] += -F_tot(r, v, self.T)
            else:               # pbc, over the walls
                r = border_dist(self.sys[pair[0]].pos, self.sys[pair[1]].pos, pair[2])
                v = self.sys[pair[0]].vel - self.sys[pair[1]].vel
                Ftable[pair[0]] += F_tot(r, v, self.T)
                Ftable[pair[1]] += -F_tot(r, v, self.T)
        return Ftable
        
    def evolve(self, dt, lmbda=0.5):
        """Function to evolve the whole system by a time step dt,
           and return system and force table
           TO DO: account only for non-zero forces"""
        vel_tilde = np.zeros(3)
        F = self.force_table()
        for i in range(self.N):
            self.sys[i].pos += dt*self.sys[i].vel + dt**2*F[i]/2
            self.sys[i].pos %= L   # pbc
            self.sys[i].vel += lmbda*dt*F[i]
        F2 = self.force_table()
        for i in range(self.N):
            self.sys[i].vel += dt*(F[i] + F2[i])/2
        
    def run(self, dt, Nsteps):    
        """Evolve the system for Nsteps with time step size dt"""
        t = np.arange(0.0, Nsteps*dt, dt)
        pos_matrix = []
        vel_matrix = []
        E = []
        for step in t:
            pos_matrix.append(self.pos_table())
            vel_matrix.append(self.vel_table())
            E.append(self.energy())
            self.evolve(dt)
        return np.array(pos_matrix), np.array(vel_matrix), E


#################################################
##### functions
def get_int_param(part1, part2, int_params):
    """get interaction parameter a for particles i, j
       NOW USELESS"""
    return int_params[part1.typ + part2.typ]
    
def wR(r):
    """Weight function"""
    return (1 - LA.norm(r)) if LA.norm(r) < 1 else 0.0
    
def sigma(T):
    return sqrt(2*gamma*kB*T)
    
def theta():
    return np.random.randn()
    

def F_C(r):
    """Conservative DPD force"""
    a = 20   #int_params[part1.typ + part2.typ]
    return a*wR(r)*r/LA.norm(r)
    
    
def F_D(r, v):
    """Dissipative DPD force, FD = -gamma wD(r) (rnorm*v) rnorm"""
    return -gamma*wR(r)**2*np.dot(r, v)*r/LA.norm(r)**2


def F_R(r, T=300):
    """Random DPD force, F^R = -sigma wR(r) theta rnorm"""
    return sigma(T)*wR(r)*theta()*r/LA.norm(r)
    
    
def F_tot(r, v, T=300):
    """Total force between two particles"""
    return F_C(r) + F_D(r, v) + F_R(r, T)
    
    
def border_dist(r1, r2, coord, L=10):
    """Get the distance of two particles 
       across border x,y,z (coord=0,1,2)"""
    shift = np.zeros(3)
    shift[coord] = L
    if r1[coord] > r2[coord]:
        return r2 + shift - r1
    else:
        return r1 + shift - r2        

def speed_list(vel_matrix):
    """Get the speed distribution averaged over time"""
    Nsteps, N, dim = vel_matrix.shape
    v = np.zeros((Nsteps, N))
    v_avg = np.zeros(N)
    for i in range(Nsteps):
        for j in range(N):
            v[i, j] = LA.norm(vel_matrix[i, j, :])
    for i in range(N):
        v_avg[i] = np.average(v[:, i])
    return v_avg

##### Old
def lookup_table(system, L=10):
    """list of pairs of particles in each other's spheres
       of interaction, accounting for PBCs with flags:
       -1: direct
        0: across x-dir wall
        1: across y-dir wall
        2: across z-dir wall
    """
    table = []
    N = len(system)
    for i in range(N):
        for j in range(i):
            r_d = LA.norm(system[i].pos - system[j].pos)
            r_x = LA.norm(border_dist(system[i].pos, system[j].pos, 0, L))
            r_y = LA.norm(border_dist(system[i].pos, system[j].pos, 1, L))
            r_z = LA.norm(border_dist(system[i].pos, system[j].pos, 2, L))
            #table.append((i, j, r_d, r_x, r_y, r_z))  # testing
            if r_d < 1:
               table.append((i, j, -1, r_d))
            elif r_x < 1:
               table.append((i, j, 0, r_x))
            elif r_y < 1:
               table.append((i, j, 1, r_y))
            elif r_z < 1:
               table.append((i, j, 2, r_z))
    return table


def force_table(system, L=10, T=300):
    """table of forces on the particles, based on lookup table"""
    ltable = lookup_table(system, L)
    Ftable = np.zeros((len(system), 3))
    for pair in ltable:
        if pair[2] == -1:   # direct
            r = system[pair[0]].pos - system[pair[1]].pos
            v = system[pair[0]].vel - system[pair[1]].vel
            Ftable[pair[0]] += F_tot(r, v, T)
            Ftable[pair[1]] += -F_tot(r, v, T)
        else:               # over the walls
            r = border_dist(system[pair[0]].pos, system[pair[1]].pos, pair[2])
            v = system[pair[0]].vel - system[pair[1]].vel
            Ftable[pair[0]] += F_tot(r, v, T)
            Ftable[pair[1]] += -F_tot(r, v, T)
    return Ftable
    
    
##### initialise and evolve the system
def initialise(N, L=10, T=300):
    """Initialise a system of N particles
       in a box of size L^3"""
    system = []
    for i in range(N):
        system.append(Particle(pos=np.random.rand(3)*L, \
                               vel=np.random.randn(3)*kB*T)) # m = 1
    return system
    
    
def timestep(system, dt, L=10, T=300, lmbda=0.5):
    """Function to evolve the whole system by a time step dt,
       and return system and force table
       TO DO: account only for non-zero forces"""
    N = len(system)
    vel_tilde = np.zeros(3)
    F = force_table(system, L, T)
    for i in range(N):
        system[i].pos = (system[i].pos + dt*system[i].vel + dt**2*F[i]/2) % L
        system[i].vel += lmbda*dt*F[i]
    F2 = force_table(system, L, T)
    for i in range(N):
        system[i].vel += dt*(F[i] + F2[i])/2
#        print system[i].vel
#    print "\n"
    return system
    
    
def run(system, Nsteps, dt, L=10, T=300):
    """Evolve the system for Nsteps with time step size dt"""
    t = np.arange(0, Nsteps*dt, dt)
    pos_matrix = []
    vel_matrix = []
    for step in t:
        print get_vel(system)
        pos_matrix.append(get_pos(system))  # WRONG APPEND!
        vel_matrix.append(get_vel(system))  # WRONG APPEND!
        system = timestep(system, dt, L, T)
    return pos_matrix, vel_matrix
    
    
##### extraction functions
def get_vel(system):
    vel = []
    for i in range(len(system)):
        vel.append(system[i].vel)
    return vel

def get_pos(system):
    pos = []
    for i in range(len(system)):
        pos.append(system[i].pos)
    return pos
    
def measure_energy(vel_matrix):
    E = []
    Nsteps = len(vel_matrix)
    for i in range(Nsteps):
        E.append(LA.norm(vel_matrix[i])**2)
    return E
    
    
          
if __name__ == "__main__":
    np.random.seed(0)
    #int_params = {"AA": 21, "AB": 21, "BA": 21}
    T = 1; N = 10; L = 5
    Nsteps = 100; dt = 1e-5
    t = np.arange(0.0, Nsteps*dt, dt)
    
    ##### testing old configuration
    #system = initialise(N, L, T)
    #tbl = lookup_table(system, L)
    #ftbl = force_table(system, L, T)
    #posm, velm = run(system, Nsteps, dt, L, T)
    
    #A = np.array(posm)
    #B = np.array(velm)    
    #E = measure_energy(velm)
    
    ##### testing system class, WORKS
    s = System(N, L, T)
    s.random_init()
    posm, velm, E = s.run(dt, Nsteps)
    
    #plt.plot(t, E)
    #plt.show()
    
    #V = speed_list(velm)
    #plt.hist(V)
    
    
        
    
    
    
    
