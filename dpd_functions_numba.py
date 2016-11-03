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
        rn += ri * ri
    return sqrt(rn)


@jit(float64(float64[:], float64[:]), nopython=True)
def dot_numba(r, v):
    rv = 0.0
    for i in range(len(r)):
        rv += r[i] * v[i]
    return rv


@jit(float64(float64[:], float64), nopython=True)
def wR(r, rc):
    """Weight function, r>0 is mutual distance"""
    nr = norm_numba(r)
    return (1 - nr / rc) if nr / rc < 1.0 else 0.0


@jit(float64[:](float64[:], float64[:],\
     float64, float64, float64, float64, float64), nopython=True)
def F_tot(r, v, a, gamma, kT, dt, rc):
    """Total force between two particles"""
    nr = norm_numba(r)
    ftot = a * wR(r, rc) * r/nr \
           - gamma * wR(r, rc)**2 * dot_numba(r, v) * r/nr**2 \
           + sqrt(2*gamma*kT) * wR(r, rc) * randn() / sqrt(dt) * r/nr
    return ftot


@jit(float64(float64[:, :], float64[:, :], int64[:], float64))#, nopython=True)
def tot_PE(X, iparams, blist, rc):
    print("", end="") # WTF? FIX THIS
    N = len(X)
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            E += iparams[blist[i], blist[j]] / 2 *\
                 (1 - norm_numba(X[i] - X[j]) / rc)**2
    return E


@jit(float64(float64[:, :]), nopython=True)
def tot_KE(V):
    """Total kinetic energy of the system,
    same mass assumed"""
    KE = 0.0
    for i in range(len(V)):
        for j in range(3):
            KE += (V[i, j] * V[i, j]) / 2.0
    return KE


@jit(float64(float64[:, :]), nopython=True)
def temperature(V):
    Ndof = len(V) # - 6  # Number of degrees of freedom, NOT SURE, FIX!
    return tot_KE(V)/(3. / 2 * Ndof)


@jit(float64[:, :](int64, float64, int64))
def init_pos(N, L, seed):
    np.random.seed(seed)
    X = rand(N, 3) * L
    return X


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
            c[i] += A[i, j] * b[j]
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
def force_list_inner(X, V, iparams, blist, L, gamma, kT, dt, rc):
    """Force matrix. Input:
    * X: (N, 3) xyz matrix
    * V: (N, 3) velocity matrix
    * iparams: (Nbt, Nbt) matrix of interaction params
    * blist: (N) list of bead types
    Output:
    * (N, N, 3) matrix"""
    N = len(X)
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
            dr = X[i] - X[j]       # rij = ri - rj
            g = np.dot(inv_cell, dr)
#            g_n = round_numba(g)
            g_n = g - np.round(g)
            dr_n = np.dot(cell, g_n)
            v_ij = V[i] - V[j]     # vij = vi - vj

#            for k in range(3):
#                dr[k] = X[i, k] - X[j, k]
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
#                v_ij[k] = V[i, k] - V[j, k]

            force_cube[i, j, :] = \
                F_tot(dr_n, v_ij, iparams[blist[i], blist[j]], gamma, kT, dt, rc)
#    force_cube -= np.transpose(force_cube, (1, 0, 2))
#    return np.sum(force_cube, axis=1)
    return force_cube


def force_list(X, V, iparams, blist, L, gamma, kT, dt, rc):
    """Wrapper for numba routine force_list_inner"""
    force_cube = force_list_inner(X, V, iparams, blist, \
                                  L, gamma, kT, dt, rc)
    force_cube -= np.transpose(force_cube, (1, 0, 2))
    return np.sum(force_cube, axis=1)


def vel_verlet_step(X, V, iparams, blist, sp):
    """The velocity Verlet algorithm. Retur:
    * X: (N, 3) matrix
    * V: (N, 3) matrix
    * number of passes through the walls"""
    F1 = force_list(X, V, iparams, blist, \
                    sp.L, sp.gamma, sp.kT, sp.dt, sp.rc)
    X2 = X + V * sp.dt + F1 * sp.dt**2 / 2.0
    V_temp = V + F1 * sp.dt / 2.0
    F2 = force_list(X2, V_temp, iparams, blist, \
                    sp.L, sp.gamma, sp.kT, sp.dt, sp.rc)
    V2 = V + (F1 + F2) * sp.dt / 2
    Npass = np.sum(X2 - X2 % sp.L != 0, 1)
    X2 = X2 % sp.L
    return X2, V2, Npass


def integrate_verlet(X, V, iparams, blist, sp):
    """
    Verlet integration for Nt steps.
    Save each thermo-multiple step into xyz_frames.
    Mass set to 1.0. Input:
    * X: (N, 3) matrix
    * V: (N, 3) matrix
    * iparams: dict mapping bead type to a_ij
    * blist: list of bead types (bead list)
    * sp: system params
    """
    T, E = np.zeros(sp.Nt), np.zeros(sp.Nt)
    ti = time.time()

    # 1st Verlet step
    F = force_list(X, V, iparams, blist, \
            sp.L, sp.gamma, sp.kT, sp.dt, sp.rc)
    X = X + V * sp.dt + F * sp.dt**2 / 2
    T[0] = temperature(V)
    E[0] = tot_KE(V) + tot_PE(X, iparams, blist, sp.rc)
    save_xyzmatrix("Dump/dump_%i.xyz" % 0, blist, X)
    tf = time.time()
    print("Step: %i | T: %.5f | Time: %.2f" % (1, T[0], tf - ti))

    # Remaining steps
    for i in range(1, sp.Nt):
        X, V, Npass = vel_verlet_step(X, V, iparams, blist, sp)
        T[i] = temperature(V)
        E[i] = tot_KE(V) + tot_PE(X, iparams, blist, sp.rc)
        tf = time.time()
        if (i+1) % sp.thermo == 0:
            save_xyzmatrix("Dump/dump_%3i.xyz" % (i+1), blist, X)
            print("Step: %i | T: %.5f | Time: %.2f" % (i+1, T[i], tf - ti))
    return T, E


def integrate_euler(X, V, iparams, blist, sp):
    """
    Euler integration for Nt steps.
    Save each thermo-multiple step into xyz_frames.
    Mass set to 1.0. Input:
    * X: (N, 3) matrix
    * V: (N, 3) matrix
    * iparams: dict mapping bead type to a_ij
    * blist: list of bead types (bead list)
    * sp: system params
    """
    T, E = np.zeros(sp.Nt), np.zeros(sp.Nt)
    ti = time.time()
    
    for i in range(sp.Nt):
        F = force_list(X, V, iparams, blist, \
                sp.L, sp.gamma, sp.kT, sp.dt, sp.rc)
        V = V + F * sp.dt
        X = X + V * sp.dt
        Npass = np.sum(X - X % sp.L != 0, 1)
        X = X % sp.L
        T[i] = temperature(V)
        E[i] = tot_KE(V) + tot_PE(X, iparams, blist, sp.rc)
        tf = time.time()
        if (i+1) % sp.thermo == 0:
            save_xyzmatrix("Dump/dump_%3i.xyz" % (i+1), blist, X)
            print("Step: %i | T: %.5f | Time: %.2f" % (i+1, T[i], tf - ti))
    return T, E


# ===== lookup table implementation
def build_lookup_table(X, L, cutoff=2.0):
    """Create a dict for each bead storing positions of 
    neighbouring beads within given cutoff
    * X: matrix of positions (N, 3)"""
    N = len(X)
    lt = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i):
            dr = X[i] - X[j]
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            if norm(dr_n) < cutoff:
                lt[i, j] = True
    return lt + lt.T


def force_list_lookup(X, V, lt, iparams, blist, sp):
    """Get force matrix from lookup table"""
    N = len(X)
    force_cube = np.zeros((N, N, 3))
    cell = sp.L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        for j in range(i):
            if lt[i, j]:
                dr = X[i] - X[j]       # rij = ri - rj
                G = np.dot(inv_cell, dr)
                G_n = G - np.round(G)
                dr_n = np.dot(cell, G_n)
                v_ij = V[i] - V[j]     # vij = vi - vj
                force_cube[i, j, :] = \
                    F_tot(dr_n, v_ij, iparams[(blist[i], blist[j])], sp)
                
    force_cube -= np.transpose(force_cube, (1, 0, 2))
    return np.sum(force_cube, axis=1)
