#!/usr/bin/env python3
"""
A collection of functions for the DPD simulation.

13/10/17
"""
import numpy as np
from numpy import sqrt
from numpy.random import rand, randn
from numba import jit, float64, int64
from dpd_io import save_xyzfile


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


@jit(float64[:](float64[:], float64[:], float64, \
        float64, float64, float64), nopython=True)
def F_tot(r, v, a, gamma, kT, dt):
    """Total force between two particles"""
    nr = norm_numba(r)
    ftot = a * wr(nr) * r / nr \
            - gamma * wr(nr)**2 * dot_numba(r, v) * r / nr**2 \
            + sqrt(2.0 * gamma * kT) * wr(nr) * randn() / sqrt(dt) * r / nr
    return ftot


@jit(float64[:, :](float64[:, :], float64[:, :], int64[:], float64[:, :], \
        float64[:, :], float64, float64, float64))#, nopython=True)
def force_mat(X, V, bl, ip, box, gamma, kT, dt):
    """Force matrix. Input:
    * X: (N, 3) xyz matrix
    * V: (N, 3) velocity matrix
    * bl: (N) list of bead types
    * ip: (Nbt, Nbt) matrix of interaction params
    Output:
    * (N, 3) force on each particle"""
    N = len(X)
    F = np.zeros((N, 3))
    Fm = np.zeros((N, N, 3))
    inv_box = np.zeros((3, 3))
    for i in range(3): inv_box[i, i] = 1.0 / box[i, i]
    g = np.zeros(3)
    rij = np.zeros(3)
    vij = np.zeros(3)

    for i in range(N):
        for j in range(i):
            rij = X[i] - X[j]
            g = matvecmul(inv_box, rij)
            g = g - round_numba(g)
#            g = g - np.round_(g, 0, np.empty_like(g))
            rij = matvecmul(box, g)
            vij = V[i] - V[j]
            Fm[i, j, :] = F_tot(rij, vij, ip[bl[i], bl[j]], gamma, kT, dt)
            Fm[j, i, :] = -Fm[i, j, :]
    F = np.sum(Fm, 1)
#    F = sum_numba(Fm)
    return F


@jit(float64(float64[:, :], int64[:], float64[:, :], float64[:, :]), \
        nopython=True)
def tot_PE(X, bl, ip, box):
    """Using '@', np.dot, inv or pinv will throw
    Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.
    """
    N = len(X)
    inv_box = np.zeros((3, 3))
#    inv_box = np.linalg.inv(box)
    for i in range(3): inv_box[i, i] = 1.0 / box[i, i]
    rij = np.zeros(3)
    g = np.zeros(3)

    E = 0.0
    for i in range(N):
        for j in range(i):
            rij = X[i] - X[j]
            g = matvecmul(inv_box, rij)
            g = g - round_numba(g)
            rij = matvecmul(box, g)
            E += ip[bl[i]-1, bl[j]-1] * wr(norm_numba(rij))**2 / 2.0
    return E


@jit(float64(float64[:, :]), nopython=True)
def tot_KE(V):
    KE = 0.0
    for i in range(len(V)):
        for j in range(3):
            KE += V[i, j]**2 / 2.0
    return KE


def euler_step(X, V, bl, ip, box, gamma, kT, dt):
    F = np.zeros(X.shape)
    F = force_mat(X, V, bl, ip, box, gamma, kT, dt)
    V += F * dt
    X += V * dt
    return X, V
 

def verlet_step(X, V, F, bl, ip, box, gamma, kT, dt):
    V2 = np.zeros(X.shape)
    V2 = V + 0.5 * F * dt
    X += V2 * dt
    F = force_mat(X, V2, bl, ip, box, gamma, kT, dt)
    V = V2 + 0.5 * F * dt
    return X, V, F


@jit(float64(float64[:, :]), nopython=True)
def temperature(V):
    return tot_KE(V) / ((3 * len(V) - 3) / 2.0)


@jit(float64[:, :](int64, float64[:, :]))
def init_pos(N, box):
    X = rand(N, 3) * np.diag(box)
    return X


@jit(float64[:, :](int64, float64))
def init_vel(N, kT):
    V = randn(N, 3) * kT
    return V - np.sum(V, 0) / N


