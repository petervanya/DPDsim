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
from dpd_io import save_xyzfile
import time


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
    N = len(g)
    gr = np.zeros(N)
    for i in range(N):
        gr[i] = g[i] - round(g[i])
    return gr


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
    return (1 - nr) if nr < 1.0 else 0.0


@jit(float64[:](float64[:], float64[:],\
     float64, float64, float64, float64, float64), nopython=True)
def F_tot(r, v, a, gamma, kT, dt):
    """Total force between two particles"""
    nr = norm_numba(r)
    ftot = a * wR(r) * r/nr \
           - gamma * wR(r)**2 * dot_numba(r, v) * r / nr**2 \
           + sqrt(2.0 * gamma * kT) * wR(r) * randn() / sqrt(dt) * r / nr
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
    fmat = np.zeros((N, 3))
    fcube = np.zeros((N, N, 3))
    inv_box = pinv(box)
    g = np.zeros(3)
    rij = np.zeros(3)
    vij = np.zeros(3)
    for i in range(N):
        for j in range(i):
            rij = X[i] - X[j]
            g = np.dot(inv_box, rij)
            g = g - np.round(g)
            rij = np.dot(box, g)
            v_ij = V[i] - V[j]

            fcube[i, j, :] = F_tot(rij, vij, ip[bl[i], bl[j]], gamma, kT, dt)
            fcube[j, i, :] = -fcube[i, j, :]
    fmat = np.sum(fcube, 2)
    return fmat


@jit(float64(float64[:, :], float64[:, :], int64[:], float64))#, nopython=True)
def tot_PE(X, bl, ip, box):
    N = len(X)
    inv_box = pinv(box)
    rij = np.zeros(3)
    g = np.zeros(3)

    E = 0.0
    for i in range(N):
        for j in range(i):
            rij = X[i] - X[j]
            g = inv_box @ rij
            g = g - np.round(g)
            rij = box @ g
            E += ip[bl[i], bl[j]] * (1 - norm_numba(rij))**2 / 2.0
    return E


@jit(float64(float64[:, :]), nopython=True)
def tot_KE(V):
    KE = 0.0
    for i in range(len(V)):
        for j in range(3):
            KE += V[i, j]**2 / 2.0
    return KE


def euler_step(X, V, bl, ip, box, gama, kT, dt):
    F = np.zeros((N, 3))
    F = force_mat(X, V2, bl, ip, box, gamma, kT, dt)
    V += F * dt
    X += V * dt
    return X, V
 

def verlet_step(X, V, F, bl, ip, box, gama, kT, dt):
    V2 = np.zeros((N, 3))
    V2 = V + 0.5 * F * dt
    X += V2 * dt
    F = force_mat(X, V2, bl, ip, box, gamma, kT, dt)
    V = V2 + 0.5 * F * dt
    return X, V, F


def integrate(X, V, bl, ip, box, gamma, kT, dt,\
        Nt, Neq, thermo, df, style="euler"):
    N = len(X)
    T = np.zeros(Nt+1)
    KE = np.zeros(Nt+1)
    PE = np.zeros(Nt+1)

    ti = time.time()    
    for i in range(2, Nt+1):
        if style == "euler":
            X, V = euler_step(X, V, bl, ip, box, gama, kT, dt)
        elif style == "verlet":
            X, V, F = verlet_step(X, V, F, bl, ip, box, gama, kT, dt)

        X = X % np.diag(box)
        KE[i] = tot_KE(V)
        PE[i] = tot_PE(X, bl, ip, box)
        T[i] = KE[i] / (3.0 / 2.0 * N)
        if i % thermo == 0:
            print("Step: %3.i | T: %.5f | KE: %.3e | PE: %.3e" % \
                (i+1, T[i], KE[i], PE[i]))
        if i >= Neq and i % df == 0:
            save_xyzfile("Dump/dump_%i.xyz" % i, bl, X)
    return T, KE, PE


@jit(float64(float64[:, :]), nopython=True)
def temperature(V):
    return tot_KE(V) / ((3 * len(V) - 3) / 2.0)


@jit(float64[:, :](int64, float64, int64))
def init_pos(N, box):
    X = rand(N, 3) * np.diag(box)
    return X


@jit(float64[:, :](float64, float64))
def init_vel(N, kT):
    V = randn(N, 3) * kT
    return V - np.sum(V, 0) / N


