#!/usr/bin/env python3
"""
A collection of functions for the DPD simulation.

23/06/16
"""
import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from dpd_io import save_xyzmatrix
import time


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


def F_tot(r, v, a, sp):
    """Total force between two particles"""
    return F_C(r, a) + F_D(r, v) + F_R(r, sp)


def V_DPD(norm_r, inter_params, sp):
    """Conservative potential energy between two beads"""
    pass


def tot_PE(pos_list, iparams, blist, sp):
    """ MAKE THIS MORE EFFICIENT """ # FINISH
    E = 0.0
    N = pos_list.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            E += iparams[(blist[i], blist[j])]/2 *\
                 (1 - norm(pos_list[i] - pos_list[j])/sp.rc)**2
    return E


def tot_KE(vel_list):
    """Total kinetic energy of the system,
    same mass assumed"""
    return np.sum(vel_list * vel_list) / 2


def init_pos(N, iparams, blist, sp):
    np.random.seed(sp.seed)
    pos_list = np.random.rand(N, 3) * sp.L
    E = tot_PE(pos_list, iparams, blist, sp)
    return pos_list, E


def init_vel(N, kT):
    """Initialise velocities"""
    return np.random.randn(N, 3) * kT


def temperature(vel_list):
    Ndof = len(vel_list)-6  # Number of degrees of freedom, NOT SURE, FIX!
    return tot_KE(vel_list)/(3./2*Ndof)


def force_list(pos_list, vel_list, iparams, blist, sp):
    """Force matrix. Input:
    * pos_list: (N, 3) xyz matrix
    * iparams: (Nbt, Nbt) matrix with interaction params
    * blist: list of bead types"""
    N = pos_list.shape[0]
    force_mat = np.zeros((N, N, 3))
    cell = sp.L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        for j in range(i):
            dr = pos_list[i] - pos_list[j]       # rij = ri - rj
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            v_ij = vel_list[i] - vel_list[j]     # vij = vi - vj
            force_mat[i, j] = F_tot(dr_n, v_ij, iparams[(blist[i], blist[j])], sp)

    force_mat -= np.transpose(force_mat, (1, 0, 2))
    return np.sum(force_mat, axis=1)


def vel_verlet_step(pos_list, vel_list, iparams, blist, sp):
    """The velocity Verlet algorithm. Retur:
    * position matrix
    * velocity matrix
    * number of passes through the walls"""
    F = force_list(pos_list, vel_list, iparams, blist, sp)
    pos_list2 = pos_list + vel_list*sp.dt + F*sp.dt**2 / 2
    F2 = force_list(pos_list2, vel_list, iparams, blist, sp)  # CHECK CORRECTNESS of vel_list
    vel_list2 = vel_list + (F + F2) * sp.dt / 2
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
    * iparams: (Nbt, Nbt) matrix with interaction params
    * blist: list of bead types (bead list)
    * sp: misc system params
    """
    T, E = np.zeros(sp.Nt), np.zeros(sp.Nt)
    ti = time.time()

    # 1st Verlet step
    F = force_list(pos_list, vel_list, iparams, blist, sp)
    pos_list = pos_list + vel_list * sp.dt + F * sp.dt**2 / 2
    T[0] = temperature(vel_list)
    if sp.saveE:
        E[0] = tot_KE(vel_list) + tot_PE(pos_list, iparams, blist, sp)
    save_xyzmatrix("Dump/dump_%i.xyz" % 0, blist, pos_list)
    print("Step: %i | T: %.2f | Time: %.2f" % (1, T[0], time.time()-ti))

    # Other steps
    for i in range(1, sp.Nt):
        pos_list, vel_list, Npass = vel_verlet_step(pos_list, vel_list, iparams, blist, sp)
        if sp.saveE:
            E[i] = tot_KE(vel_list) + tot_PE(pos_list, iparams, blist, sp)
        T[i] = temperature(vel_list)
        if (i+1) % sp.thermo == 0:
            save_xyzmatrix("Dump/dump_%i.xyz" % (i+1), blist, pos_list)
            print("Step: %i | T: %.2f | Time: %.2f" % (i+1, T[i], time.time()-ti))
    return T, E


