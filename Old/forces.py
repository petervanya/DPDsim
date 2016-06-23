#!/usr/bin/env python
"""
Module containing DPD forces

07/07/2015
"""
import numpy as np

def get_int_param(part1, part2, int_params):
    """get interaction parameter a for particles i,j
       NOW USELESS"""
    return int_params[part1.typ + part2.typ]
    
def wR(r):
    return (1 - LA.norm(r)) if LA.norm(r) < 1 else 0.0
    
def sigma(T, gamma=1):
    return sqrt(2*gamma*kB*T)
    
def theta():
    return np.random.randn()
    
    
##### dynamics, forces
def F_C(part1, part2, shift=np.array([0, 0, 0])):
    """conservative DPD force"""
    r = part1.pos - part2.pos
    a = 20   #int_params[part1.typ + part2.typ]
    return a*wR(r)*r/LA.norm(r)
    
    
def F_D(part1, part2, gamma=1):
    """dissipative DPD force, FD = -gamma wD(r) (rnorm*v) rnorm"""
    r = part1.pos - part2.pos
    v = part1.vel - part2.vel
    return -gamma*wR(r)**2*np.dot(r, v)*r/LA.norm(r)**2


def F_R(part1, part2, T=300):
    """random DPD force, F^R = -sigma wR(r) theta rnorm"""
    r = part1.pos - part2.pos
    return sigma(T)*wR(r)*theta()*r/LA.norm(r)
    
    
def F_tot(part1, part2, T=300, gamma=1):
    """total force between two particles"""
    force = F_C(part1, part2) + F_D(part1, part2, gamma) + \
            F_R(part1, part2, T)
    return force
    
    
#### Old
#def force_matrix(system, T=300, gamma=1):
#    """the matrix of forces
#       NOW OBSOLETE"""
#    N = len(system)
#    F = np.zeros((N, N, 3))
#    for i in range(N):
#        for j in range(i):
#             F[i, j, :] = F_C(system[i], system[j]) + \
#                       F_D(system[i], system[j], gamma) + \
#                       F_R(system[i], system[j], T)
#             F[j, i, :] = -F[i, j, :]
#    return F
#    
    
