#!/usr/bin/env python3
"""
A collection of functions for the DPD simulation.

23/06/16
"""
import numpy as np
from numpy import sqrt
from numpy.random import rand, randn
from numpy.linalg import norm
from numba import jit, float64, int64
from dpd_io import save_xyzmatrix
import time


@jit(float64(float64[:]), nopython=True)
def norm_numba(r):
    rn = 0.0
    for ri in r:
        rn += ri*ri
    return sqrt(rn)


@jit(float64(float64[:], float64[:]), nopython=True)
def dot_numba(r, v):
    rv = 0.0
    for i in range(len(r)):
        rv += r[i]*v[i]
    return rv


@jit(float64(float64[:], float64), nopython=True)
def wR(r, rc):
    """Weight function, r>0 is mutual distance"""
    nr = norm_numba(r)
    return (1 - nr/rc) if nr/rc < 1.0 else 0.0


@jit(float64[:](float64[:], float64[:],\
     float64, float64, float64, float64, float64), nopython=True)
def F_tot(r, v, a, gamma, kT, dt, rc):
    """Total force between two particles"""
    nr = norm_numba(r)
    ftot = a * wR(r, rc) * r/nr \
           - gamma * wR(r, rc)**2 * dot_numba(r, v) * r/nr**2 \
           + sqrt(2*gamma*kT) * wR(r, rc) * randn() / sqrt(dt) * r/nr
    return ftot


@jit(float64(float64[:, :], float64[:, :], int64[:], float64), nopython=True)
def tot_PE(pos_list, iparams, blist, rc):
    E = 0.0
    N = len(pos_list)
    for i in range(N):
        for j in range(i+1, N):
            E += iparams[blist[i], blist[j]]/2 *\
                 (1 - norm_numba(pos_list[i] - pos_list[j])/rc)**2
    return E


@jit(float64(float64[:, :]), nopython=True)
def tot_KE(vel_list):
    """Total kinetic energy of the system,
    same mass assumed"""
    KE = 0.0
    for i in range(len(vel_list)):
        for j in range(3):
            KE += (vel_list[i, j]*vel_list[i, j]) / 2.0
    return KE


@jit(float64(float64[:, :]), nopython=True)
def temperature(vel_list):
    Ndof = len(vel_list)# - 6  # Number of degrees of freedom, NOT SURE, FIX!
    return tot_KE(vel_list)/(3./2*Ndof)


@jit(float64[:, :](int64, float64, int64))
def init_pos(N, L, seed):
    np.random.seed(seed)
    pos_list = rand(N, 3) * L
    return pos_list


@jit(float64[:, :](float64, float64))
def init_vel(N, kT):
    """Initialise velocities"""
    return randn(N, 3) * kT


@jit(float64[:](float64[:, :], float64[:]), nopython=True)
def matvecmul(A, b):
    N, M = A.shape
    c = np.zeros(N)
    for i in range(N):
        for j in range(M):
            c[i] += A[i, j]*b[j]
    return c


@jit(float64[:](float64[:]))
def round_numba(g):
    N = len(g)
    gr = np.zeros(N)
    for i in range(N):
        gr[i] = g[i] - round(g[i])
    return gr


@jit(float64[:, :, :](float64[:, :], float64[:, :], float64[:, :], int64[:], \
     float64, float64, float64, float64, float64))#, nopython=True)
def force_list_inner(pos_list, vel_list, iparams, blist, L, gamma, kT, dt, rc):
    """Force matrix. Input:
    * pos_list: (N, 3) xyz matrix
    * vel_list: (N, 3) velocity matrix
    * iparams: (Nbt, Nbt) matrix of interaction params
    * blist: (N) list of bead types
    Output:
    * (N, N, 3) matrix"""
    N = len(pos_list)
    force_cube = np.zeros((N, N, 3))
    cell = L*np.eye(3)
    inv_cell = np.linalg.inv(cell)
    g = np.zeros(3)
    g_n = np.zeros(3)
    g_rounded = np.zeros(3)
    dr = np.zeros(3)
    dr_n = np.zeros(3)
    v_ij = np.zeros(3)
    for i in range(N):
        for j in range(i):
            dr = pos_list[i] - pos_list[j]       # rij = ri - rj
            g = np.dot(inv_cell, dr)
#            g_n = round_numba(g)
            g_n = g - np.round(g)
            dr_n = np.dot(cell, g_n)
            v_ij = vel_list[i] - vel_list[j]     # vij = vi - vj

#            for k in range(3):
#                dr[k] = pos_list[i, k] - pos_list[j, k]
#            for k in range(3):
#                g[k] = 0.
#                for l in range(3):
#                    g[k] += inv_cell[k, l]*dr[l]
#            for k in range(3):
#                g_rounded[k] = round(g[k])
#            for k in range(3):
#                g_n[k] = g[k] - g_rounded[k]
#            for k in range(3):
#                dr_n[k] = 0.
#                for l in range(3):
#                    dr_n[k] += cell[k, l]*g_n[l]
#            for k in range(3):
#                v_ij[k] = vel_list[i, k] - vel_list[j, k]

            force_cube[i, j, :] = \
                F_tot(dr_n, v_ij, iparams[blist[i], blist[j]], gamma, kT, dt, rc)
#    force_cube -= np.transpose(force_cube, (1, 0, 2))
#    return np.sum(force_cube, axis=1)
    return force_cube


def force_list(pos_list, vel_list, iparams, blist, L, gamma, kT, dt, rc):
    """Wrapper for numba routine force_list_inner"""
    force_cube = force_list_inner(pos_list, vel_list, iparams, blist, \
                                  L, gamma, kT, dt, rc)
    force_cube -= np.transpose(force_cube, (1, 0, 2))
    return np.sum(force_cube, axis=1)


def vel_verlet_step(pos_list, vel_list, iparams, blist, sp):
    """The velocity Verlet algorithm. Retur:
    * pos_list: (N, 3) matrix
    * vel_list: (N, 3) matrix
    * number of passes through the walls"""
    F1 = force_list(pos_list, vel_list, iparams, blist, \
                    sp.L, sp.gamma, sp.kT, sp.dt, sp.rc)
    pos_list2 = pos_list + vel_list * sp.dt + F1 * sp.dt**2 / 2.0
    vel_list_temp = vel_list + F1 * sp.dt / 2.0
    F2 = force_list(pos_list2, vel_list_temp, iparams, blist, \
                    sp.L, sp.gamma, sp.kT, sp.dt, sp.rc)
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

    ti = time.time()
    # 1st Verlet step
    F = force_list(pos_list, vel_list, iparams, blist, sp.L, sp.gamma, sp.kT, sp.dt, sp. rc)
    pos_list = pos_list + vel_list * sp.dt + F * sp.dt**2 / 2
    T[0] = temperature(vel_list)
    if sp.saveE:
        E[0] = tot_KE(vel_list) + tot_PE(pos_list, iparams, blist, rc)
    save_xyzmatrix("Dump/dump_%i.xyz" % 0, blist, pos_list)
    print("Step: %i | T: %.2f | Time: %.2f" % (1, T[0], time.time()-ti))

    # Other steps
    for i in range(1, sp.Nt):
        pos_list, vel_list, Npass = vel_verlet_step(pos_list, vel_list, iparams, blist, sp)
        if sp.saveE:
            E[i] = tot_KE(vel_list) + tot_PE(pos_list, iparams, blist, rc)
        T[i] = temperature(vel_list)
        if (i+1) % sp.thermo == 0:
            save_xyzmatrix("Dump/dump_%i.xyz" % (i+1), blist, pos_list)
            print("Step: %i | T: %.2f | Time: %.2f" % (i+1, T[i], time.time()-ti))
    return T, E


# ===== lookup table implementation
def build_lookup_table(pos_list, L, cutoff=2.0)
    """Create a dict for each bead storing positions of 
    neighbouring beads within given cutoff"""
    N = len(pos_list)
    lt = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i):
            dr = pos_list[i] - pos_list[j]
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            if norm(dr_n) < cutoff:
                lt[i, j] = True
    return lt + lt.T


def force_list_lookup(pos_list, vel_list, lt, iparams, blist, sp):
    """Get force matrix from lookup table"""
    N = len(pos_list)
    force_cube = np.zeros((N, N, 3))
    cell = sp.L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        for j in range(i):
            if lt[i, j]:
                dr = pos_list[i] - pos_list[j]       # rij = ri - rj
                G = np.dot(inv_cell, dr)
                G_n = G - np.round(G)
                dr_n = np.dot(cell, G_n)
                v_ij = vel_list[i] - vel_list[j]     # vij = vi - vj
                force_cube[i, j, :] = \
                    F_tot(dr_n, v_ij, iparams[(blist[i], blist[j])], sp)
                
    force_cube -= np.transpose(force_cube, (1, 0, 2))
    return np.sum(force_cube, axis=1)
