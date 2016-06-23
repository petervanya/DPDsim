#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from math import sqrt
#from forces import F_tot

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
            E += norm(self[i].vel)**2/2
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
                r_d = norm(self.sys[i].pos - self.sys[j].pos)
                r_x = norm(border_dist(self.sys[i].pos, self.sys[j].pos, 0, self.L))
                r_y = norm(border_dist(self.sys[i].pos, self.sys[j].pos, 1, self.L))
                r_z = norm(border_dist(self.sys[i].pos, self.sys[j].pos, 2, self.L))
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


def border_dist(r1, r2, coord, L=10):
    """Get the distance of two particles 
       across border x,y,z (coord=0,1,2)"""
    shift = np.zeros(3)
    shift[coord] = L
    if r1[coord] > r2[coord]:
        return r2 + shift - r1
    else:
        return r1 + shift - r2        



def get_int_param(part1, part2, int_params):
    """get interaction parameter a for particles i,j
       NOW USELESS"""
    return int_params[part1.typ + part2.typ]
    
def wR(r):
    return (1 - norm(r)) if norm(r) < 1 else 0.0
    
def sigma(T, gamma=1):
    return sqrt(2*gamma*kB*T)
    
def theta():
    return np.random.randn()
    
    
##### dynamics, forces
def F_C(part1, part2, shift=np.array([0, 0, 0])):
    """conservative DPD force"""
    r = part1.pos - part2.pos
    a = 20   #int_params[part1.typ + part2.typ]
    return a*wR(r)*r/norm(r)
    
    
def F_D(part1, part2, gamma=1):
    """dissipative DPD force, FD = -gamma wD(r) (rnorm*v) rnorm"""
    r = part1.pos - part2.pos
    v = part1.vel - part2.vel
    return -gamma*wR(r)**2*np.dot(r, v)*r/norm(r)**2


def F_R(part1, part2, T=300):
    """random DPD force, F^R = -sigma wR(r) theta rnorm"""
    r = part1.pos - part2.pos
    return sigma(T)*wR(r)*theta()*r/norm(r)
    
    
def F_tot(part1, part2, T=300, gamma=1):
    """total force between two particles"""
    force = F_C(part1, part2) + F_D(part1, part2, gamma) + \
            F_R(part1, part2, T)
    return force
