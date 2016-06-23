#!/usr/bin/env python
"""
Module containing DPD forces

07/07/2015
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import sqrt

gamma = 1

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
def F_C(r):
    """conservative DPD force"""
    a = 20   #int_params[part1.typ + part2.typ]
    return a*wR(r)*r/LA.norm(r)
    
    
def F_D(r, v):
    """dissipative DPD force, FD = -gamma wD(r) (rnorm*v) rnorm"""
    return -gamma*wR(r)**2*np.dot(r, v)*r/LA.norm(r)**2


def F_R(r, T=300):
    """random DPD force, F^R = -sigma wR(r) theta rnorm"""
    return sigma(T)*wR(r)*theta()*r/LA.norm(r)
    
    
def F_tot(r, v, T=300):
    """total force between two particles"""
    return F_C(r) + F_D(r, v) + F_R(r, T)
    
    
    
