#!/usr/bin/env python3
"""
A collection of functions for the DPD simulation.

23/06/16
"""
import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from numba import jit, float64, int64
from dpd_io import save_xyzmatrix


def wR(r, rc=1.0):
    """Weight function"""
    return (1 - norm(r)/rc) if norm(r) < rc else 0.0


def theta(dt):
    return np.random.randn()/sqrt(dt)
    

def F_C(r, a=25.0):
    """Conservative DPD force"""
    return a*wR(r)*r/norm(r)
    
    
def F_D(r, v, gamma=4.5):
    """Dissipative DPD force, FD = -gamma wD(r) (rnorm*v) rnorm"""
    return -gamma*wR(r)**2*np.dot(r, v)*r/norm(r)**2


def F_R(r, sp):
    """Random DPD force, F^R = -sigma wR(r) theta rnorm"""
    return sqrt(2*sp.gamma*sp.kT)*wR(r)*theta(sp.dt)*r/norm(r)


@jit(float64[:](float64[:], float64, float64, float64))
def F_R(r, gamma, kT, dt):
    """Random DPD force, F^R = -sigma wR(r) theta rnorm
    * r: vector"""
    return sqrt(2*gamma*kT)*wR(r)*theta(dt)*r/norm(r)
    
    
@jit(float64[:](float64[:], float64[:], float64, float64, float64, float64))
def F_tot(r, v, a, gamma, kT, dt):
    """Total force between two particles"""
    return F_C(r, a) + F_D(r, v) + F_R(r, gamma, kT, dt)


@jit(float64(float64[:, :], float64[:, :], int64[:], int64))
def tot_PE(pos_list, iparams, blist, rc):
    """ MAKE THIS MORE EFFICIENT """ # FINISH
    E = 0.0
    N = pos_list.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            E += iparams[blist[i], blist[j]]/2 *\
                 (1 - norm(pos_list[i] - pos_list[j])/rc)**2
    return E


@jit(float64(float64[:, :]))
def tot_KE(vel_list):
    """Total kinetic energy of the system,
    same mass assumed"""
    return np.sum(vel_list * vel_list) / 2


@jit(float64[:, :](float64, float64[:, :], int64[:], float64, float64, int64))
def init_pos(N, iparams, blist, L, rc, seed=1234):
    np.random.seed(seed)
    pos_list = np.random.rand(N, 3) * L
    return pos_list


@jit(float64[:, :](float64, float64))
def init_vel(N, kT):
    """Initialise velocities"""
    return np.random.randn(N, 3) * kT


@jit(float64(float64[:, :]))
def temperature(vel_list):
    Ndof = len(vel_list)-6  # Number of degrees of freedom, NOT SURE, FIX!
    return tot_KE(vel_list)/(3./2*Ndof)


@jit(float64[:, :](float64[:, :], float64[:, :], float64[:, :], int64[:], \
     float64, float64, float64, float64))
def force_list(pos_list, vel_list, iparams, blist, L, gamma=4.5, kT=1.0, dt=0.02):
    """Force matrix. Input:
    * pos_list: (N, 3) xyz matrix
    * vel_list: (N, 3) velocity matrix
    * iparams: (Nbt, Nbt) matrix of interaction params
    * blist: (N) list of bead types
    Output:
    * (N, 3) matrix"""
    N = len(pos_list)
    force_mat = np.zeros((N, N, 3))
    cell = L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        for j in range(i):
            dr = pos_list[i] - pos_list[j]       # rij = ri - rj
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            v_ij = vel_list[i] - vel_list[j]     # vij = vi - vj
            force_mat[i, j, :] = \
                F_tot(dr_n, v_ij, iparams[blist[i], blist[j]], gamma, kT, dt)

    force_mat -= np.transpose(force_mat, (1, 0, 2))
    return np.sum(force_mat, axis=1)


def vel_verlet_step(pos_list, vel_list, iparams, blist, sp):
    """The velocity Verlet algorithm. Retur:
    * pos_list: (N, 3) matrix
    * vel_list: (N, 3) matrix
    * number of passes through the walls"""
    F1 = force_list(pos_list, vel_list, iparams, blist, \
                    sp.L, sp.gamma, sp.kT, sp.dt)
    pos_list2 = pos_list + vel_list*sp.dt + F1 * sp.dt**2 / 2
    F2 = force_list(pos_list2, vel_list, iparams, blist, \
                    sp.L, sp.gamma, sp.kT, sp.dt)
    vel_list2 = vel_list + (F1 + F2) * sp.dt / 2
    Npass = np.sum(pos_list2 - pos_list2 % sp.L != 0, axis=1)
    pos_list2 = pos_list2 % sp.L
    return pos_list2, vel_list2, Npass


def integrate(pos_list, vel_list, iparams, blist, sp):
    """
    Verlet integration for Nt steps.
    Save each thermo-multiple step into xyz_frames.
    Mass set to 1.0. Input:
    * pos_list: (N, 3) matrix
    * vel_list: (N, 3) matrix
    * iparams: dict mapping bead type to a_ij
    * blist: list of bead types (bead list)
    * sp: misc system params
    """
    T, E = np.zeros(sp.Nt), np.zeros(sp.Nt)

    # 1st Verlet step
    F = force_list(pos_list, vel_list, iparams, blist, sp.L, sp.gamma, sp.kT, sp.dt)
    pos_list = pos_list + vel_list * sp.dt + F * sp.dt**2 / 2
    T[0] = temperature(vel_list)
    if sp.saveE:
        E[0] = tot_KE(vel_list) + tot_PE(pos_list, iparams, blist, rc)
    save_xyzmatrix("Dump/dump_%i.xyz" % 0, blist, pos_list)

    # Other steps
    for i in range(1, sp.Nt):
        pos_list, vel_list, Npass = vel_verlet_step(pos_list, vel_list, iparams, blist, sp)
        if sp.saveE:
            E[i] = tot_KE(vel_list) + tot_PE(pos_list, iparams, blist, rc)
        T[i] = temperature(vel_list)
        if (i+1) % sp.thermo == 0:
            save_xyzmatrix("Dump/dump_%i.xyz" % (i+1), blist, pos_list)
            print("Step: %i, Temperature: %f" % (i+1, T[i]))
    return T, E


