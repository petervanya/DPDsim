#!/usr/bin/env python3
"""
A collection of functions for the DPD simulation.

13/10/17
"""
import numpy as np
from numpy import sqrt
from numpy.random import rand, randn
from numba import jit, float64, int64
from .sim_io import save_xyzfile


# =====
# Numba helper functions
# =====
@jit(float64(float64[:]), nopython=True)
def norm_numba(r):
    rn = 0.0
    for ri in r:
        rn += ri * ri
    return sqrt(rn)


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
    """Does not work with nopython"""
    N = len(g)
    gr = np.zeros(N)
    for i in range(N):
        gr[i] = round(g[i])
    return gr


@jit(float64(float64[:], float64[:]), nopython=True)
def dot_numba(r, v):
    rv = 0.0
    for i in range(len(r)):
        rv += r[i] * v[i]
    return rv

@jit(float64[:, :](float64[:, :, :]), nopython=True)
def sum_numba(A):
    """Sum by 1st index"""
    N = A.shape
    B = np.zeros((N[0], N[2]))
    for i in range(N[0]):
        for j in range(N[2]):
            for k in range(N[1]):
                B[i, j] += A[i, k, j]
    return B


@jit(float64(float64), nopython=True)
def wr(nr):
    """Weight function, r>0 is mutual distance"""
    return (1 - nr) if nr < 1.0 else 0.0


#@jit(float64[:](float64[:], float64[:], float64, \
#        float64, float64, float64), nopython=True)
#def F_tot(r, v, a, gamma, kT, dt):
#    """Total force between two particles"""
#    nr = norm_numba(r)
#    ftot = a * wr(nr) * r / nr \
#            - gamma * wr(nr)**2 * dot_numba(r, v) * r / nr**2 \
#            + sqrt(2.0 * gamma * kT) * wr(nr) * randn() / sqrt(dt) * r / nr
#    return ftot


# =====
# Physics
# =====
@jit(nopython=True)
def compute_force(X, V, bl, ip, box, gamma, kT, dt):
    """
    Compute force on each particle. 
    Need full np.random.randn(), not randn(), to work with nopython.
    
    Input
    =====
    * X: (N, 3) xyz matrix
    * V: (N, 3) velocity matrix
    * bl: (N) list of bead types
    * ip: (Nbt, Nbt) matrix of interaction params
    * gamma: scalar friction
    * kT: scalar temperature
    * dt: scalar timestep
    
    Output
    ======
    * F: (N, 3) force on each particle
    * vir: the virial sum_ij Fij rij
    * sigma: (3, 3) stress tensor
    """
    N = len(X)
    F = np.zeros((N, 3))
    Fm = np.zeros((N, N, 3))
    inv_box = np.zeros((3, 3))
    for i in range(3): inv_box[i, i] = 1.0 / box[i, i]
    g = np.zeros(3)
    rij = np.zeros(3)
    vij = np.zeros(3)
    a = 0.0
    nr = 0.0
    fpair = 0.0

    vir = 0.0
    sigma_kin = np.zeros(3)
    sigma_pot = np.zeros(3)
    sigma = np.zeros(3)
    volume = np.linalg.det(box)

    for i in range(N):
        for j in range(i):
            rij = X[i] - X[j]
            g = matvecmul(inv_box, rij)
            g = g - np.round_(g, 0, np.empty_like(g))
            rij = matvecmul(box, g)
            vij = V[i] - V[j]

            a = ip[bl[i]-1, bl[j]-1]
            nr = norm_numba(rij)

            fc = a * wr(nr)
            fpair = fc \
                - gamma * wr(nr)**2 * dot_numba(rij, vij) / nr \
                + sqrt(2.0*gamma*kT) * wr(nr) * np.random.randn() / sqrt(dt)
            Fm[i, j, :] = fpair / nr * rij
            Fm[j, i, :] = -fpair / nr * rij

            vir += Fm[i, j, :] @ rij
            sigma_pot += Fm[i, j, :] * rij

    # kinetic part of stress tensor
    for i in range(N):
        sigma_kin += V[i] * V[i]

    sigma = (sigma_kin + sigma_pot) / volume
    F = np.sum(Fm, 1)

    return F, vir, sigma


@jit(float64(float64[:, :], int64[:], float64[:, :], float64[:, :]), \
        nopython=True)
def compute_pe(X, bl, ip, box):
    """Using '@', np.dot, inv or pinv will throw
    Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.
    """
    N = len(X)
    inv_box = np.zeros((3, 3))
    for i in range(3): inv_box[i, i] = 1.0 / box[i, i]
    rij = np.zeros(3)
    g = np.zeros(3)
    a = 0.0

    PE = 0.0
    for i in range(N):
        for j in range(i):
            rij = X[i] - X[j]
            g = matvecmul(inv_box, rij)
            g = g - round_numba(g)
            rij = matvecmul(box, g)
            a = ip[bl[i]-1, bl[j]-1]
            PE += a * wr(norm_numba(rij))**2 / 2.0
    return PE


@jit(float64(float64[:, :]), nopython=True)
def compute_ke(V):
    KE = 0.0
    for i in range(len(V)):
        for j in range(3):
            KE += V[i, j]**2 / 2.0
    return KE


@jit(float64(float64[:, :]), nopython=True)
def compute_temperature(V):
    return compute_ke(V) / ((3 * len(V) - 3) / 2.0)


# =====
# Initialisation of frames
# =====
@jit(float64[:, :](int64, float64[:, :]))
def init_pos(N, box):
    X = rand(N, 3) * np.diag(box)
    return X


@jit(float64[:, :](int64, float64))
def init_vel(N, kT):
    V = randn(N, 3) * kT
    return V - np.sum(V, 0) / N


# =====
# Integration steps
# =====
def euler_step(X, V, bl, ip, box, gamma, kT, dt):
    F = np.zeros(X.shape)
    F, vir, sigma = compute_force(X, V, bl, ip, box, gamma, kT, dt)
    V += F * dt
    X += V * dt
    return X, V, F, vir, sigma
 

def verlet_step(X, V, F, bl, ip, box, gamma, kT, dt):
    V2 = np.zeros(X.shape)
    V2 = V + 0.5 * F * dt
    X += V2 * dt
    F, vir, sigma = compute_force(X, V2, bl, ip, box, gamma, kT, dt)
    V = V2 + 0.5 * F * dt
    return X, V, F, vir, sigma





