#!/usr/bin/env python
"""
A collection of functions for the DPD simulation.

23/06/16
"""
import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from dpd_io import save_xyzmatrix


def wR(r, rc=1.0):
    """Weight function"""
    return (1 - norm(r)/rc) if norm(r) < rc else 0.0


def theta():
    return np.random.randn()
    

def F_C(r):  # THINK THROUGH, ADD BEAD TYPE
    """Conservative DPD force"""
    a = 25.0
    return a*wR(r)*r/norm(r)
    
    
def F_D(r, v, gamma=4.5):
    """Dissipative DPD force, FD = -gamma wD(r) (rnorm*v) rnorm"""
    return -gamma*wR(r)**2*np.dot(r, v)*r/norm(r)**2


def F_R(r, gamma, kBT=1.0):
    """Random DPD force, F^R = -sigma wR(r) theta rnorm"""
    return sqrt(2*gamma*kBT)*wR(r)*theta()*r/norm(r)
    
    
def F_tot(r, v, gamma, kBT=1.0):   # ADD BEAD TYPE
    """Total force between two particles"""
    return F_C(r) + F_D(r, v) + F_R(r, gamma, kBT)


def V_DPD(norm_r, inter_params, sp):
    """Conservative potential energy between two beads"""
    pass


def tot_PE(pos_list, int_params, sp):
    """ MAKE THIS MORE EFFICIENT """ # FINISH
    E = 0.0
    N = pos_list.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            E += int_params[(i, j)]/2 * (1 - norm(pos_list[i] - pos_list[j])/sp.rc)**2
    return E


def tot_KE(vel_list):
    """Total kinetic energy of the system,
    same mass assumed"""
    return np.sum(vel_list * vel_list) / 2


def temperature(vel_list):
    Ndof = len(vel_list)   # Number of degrees of freedom, NOT TRUE, FIX!
    return tot_KE(vel_list)/(3./2*Ndof)


def init_pos(N, int_params, sp):
    np.random.seed(sp.seed)
    pos_list = np.random.rand((N, 3)) * sp.L
    E = tot_PE(pos_list, int_params, sp)
    return pos_list, E


def init_vel(N, kBT):
    """Initialise velocities"""
    return np.random.randn(N, 3) * 3*kBT


def force_list(pos_list, sp):
    """Force matrix"""
    N = pos_list.shape[0]
    force_mat = np.zeros((N, N, 3))
    cell = sp.L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        for j in range(i):
            dr = pos_list[j] - pos_list[i]
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            force_mat[i, j] = force(dr_n, sp)

    force_mat -= np.transpose(force_mat, (1, 0, 2))
    return np.sum(force_mat, axis=1)


def vel_verlet_step(pos_list, vel_list, sp):
    """The velocity Verlet algorithm,
    returning position and velocity matrices"""
    with timing('force_list'):
        if sp.use_numba:
            F = force_list_numba(pos_list, sp.L, sp.eps, sp.sigma, sp.rc)
        elif sp.use_cython:
            F = ljc.force_list(pos_list, sp)
        elif sp.use_fortran:
            F = ljf.force_list(pos_list, sp.L, sp.eps, sp.sigma, sp.rc, np.linalg.inv)
        elif sp.use_cfortran:
            F = ljcf.force_list(pos_list, sp)
        else:
            F = force_list(pos_list, sp)
    pos_list2 = pos_list + vel_list * sp.dt + F * sp.dt**2 / 2
    with timing('force_list'):
        if sp.use_numba:
            F2 = force_list_numba(pos_list2, sp.L, sp.eps, sp.sigma, sp.rc)
        elif sp.use_cython:
            F2 = ljc.force_list(pos_list2, sp)
        elif sp.use_fortran:
            F2 = ljf.force_list(pos_list2, sp.L, sp.eps, sp.sigma, sp.rc, np.linalg.inv)
        elif sp.use_cfortran:
            F2 = ljcf.force_list(pos_list2, sp)
        else:
            F2 = force_list(pos_list2, sp)
    vel_list2 = vel_list + (F + F2) * sp.dt / 2
    Npasses = np.sum(pos_list2 - pos_list2 % sp.L != 0, axis=1)
    pos_list2 = pos_list2 % sp.L
    return pos_list2, vel_list2, Npasses


def integrate(pos_list, vel_list, sp):
    """
    Verlet integration for Nt steps.
    Save each thermo-multiple step into xyz_frames.
    Mass set to 1.0.
    """
    # N = pos_list.shape[0]
    # Nframes = int(sp.Nt // sp.thermo)
    n_fr = 1
    # xyz_frames = np.zeros((N, 3, Nframes))
    E = np.zeros(sp.Nt)
    T = np.zeros(sp.Nt)

    # 1st Verlet step
    with timing('force_list'):
        if sp.use_numba:
            F = force_list_numba(pos_list, sp.L, sp.eps, sp.sigma, sp.rc)
        elif sp.use_cython:
            F = ljc.force_list(pos_list, sp)
        elif sp.use_fortran:
            F = ljf.force_list(pos_list, sp.L, sp.eps, sp.sigma, sp.rc, np.linalg.inv)
        elif sp.use_cfortran:
            F = ljcf.force_list(pos_list, sp)
        else:
            F = force_list(pos_list, sp)
    pos_list = pos_list + vel_list * sp.dt + F * sp.dt**2 / 2
    with timing('tot_PE'):
        if sp.use_numba:
            E[0] = tot_KE(vel_list) + tot_PE_numba(pos_list, sp.eps, sp.sigma, sp.rc)
        elif sp.use_cython:
            E[0] = tot_KE(vel_list) + ljc.tot_PE(pos_list, sp)
        elif sp.use_fortran:
            E[0] = tot_KE(vel_list) + ljf.tot_pe(pos_list, sp.eps, sp.sigma, sp.rc)
        elif sp.use_cfortran:
            E[0] = tot_KE(vel_list) + ljcf.tot_PE(pos_list, sp)
        else:
            E[0] = tot_KE(vel_list) + tot_PE(pos_list, sp)
    T[0] = temperature(vel_list)

    # Other steps
    for i in range(1, sp.Nt):
        pos_list, vel_list, Npasses = vel_verlet_step(pos_list, vel_list, sp)
        with timing('tot_PE'):
            if sp.use_numba:
                E[i] = tot_KE(vel_list) + tot_PE_numba(pos_list, sp.eps, sp.sigma, sp.rc)
            elif sp.use_cython:
                E[i] = tot_KE(vel_list) + ljc.tot_PE(pos_list, sp)
            elif sp.use_fortran:
                E[i] = tot_KE(vel_list) + ljf.tot_pe(pos_list, sp.eps, sp.sigma, sp.rc)
            elif sp.use_cfortran:
                E[i] = tot_KE(vel_list) + ljcf.tot_PE(pos_list, sp)
            else:
                E[i] = tot_KE(vel_list) + tot_PE(pos_list, sp)
        T[i] = temperature(vel_list)
        if i % sp.thermo == 0:
            # xyz_frames[:, :, n_fr] = pos_list
            if sp.dump:
                fname = "Dump/dump_" + str(i*sp.thermo) + ".xyz"
                save_xyzmatrix(fname, pos_list)
            print("Step: %i, Temperature: %f" % (i, T[i]))
            n_fr += 1
    # return xyz_frames, E
    return E
