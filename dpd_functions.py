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


def tot_PE(X, iparams, blist, sp):
    """ MAKE THIS MORE EFFICIENT """ # FINISH
    E = 0.0
    N = X.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            E += iparams[(blist[i], blist[j])] / 2 *\
                 (1 - norm(X[i] - X[j]) / sp.rc)**2
    return E


def tot_KE(V):
    """Total kinetic energy of the system,
    same mass assumed"""
    return np.sum(V * V) / 2


def init_pos(N, L, seed=1234):
    np.random.seed(seed)
    X = np.random.rand(N, 3) * L
    return X


def init_vel(N, kT):
    """Initialise velocities"""
    return np.random.randn(N, 3) * kT


def temperature(V):
    Ndof = len(V) # -6  # Number of degrees of freedom, NOT SURE, FIX!
    return tot_KE(V) / (3. / 2 * Ndof)


def force_list(X, V, iparams, blist, sp):
    """Force matrix. Input:
    * X: (N, 3) xyz matrix
    * iparams: (Nbt, Nbt) matrix with interaction params
    * blist: list of bead types"""
    N = len(X)
    force_mat = np.zeros((N, N, 3))
    cell = sp.L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        for j in range(i):
            dr = X[i] - X[j]       # rij = ri - rj
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            v_ij = V[i] - V[j]     # vij = vi - vj
            force_mat[i, j] = F_tot(dr_n, v_ij, iparams[(blist[i], blist[j])], sp)

    force_mat -= np.transpose(force_mat, (1, 0, 2))
    return np.sum(force_mat, axis=1)


def vel_verlet_step(X, V, iparams, blist, sp):
    """The velocity Verlet algorithm. Retur:
    * position matrix
    * velocity matrix
    * number of passes through the walls"""
    F = force_list(X, V, iparams, blist, sp)
    X2 = X + V*sp.dt + F*sp.dt**2 / 2
    F2 = force_list(X2, V, iparams, blist, sp)  # CHECK CORRECTNESS of V
    V2 = V + (F + F2) * sp.dt / 2
    Npass = np.sum(X2 - X2 % sp.L != 0, axis=1)
    X2 = X2 % sp.L
    return X2, V2, Npass


def integrate_verlet(X, V, iparams, blist, sp):
    """
    Verlet integration for Nt steps.
    Save each thermo-multiple step into xyz_frames.
    Mass set to 1.0. Input:
    * X: (N, 3) matrix
    * V: (N, 3) matrix
    * iparams: (Nbt, Nbt) matrix with interaction params
    * blist: list of bead types (bead list)
    * sp: misc system params
    """
    T, E = np.zeros(sp.Nt), np.zeros(sp.Nt)
    ti = time.time()

    # 1st Verlet step
    F = force_list(X, V, iparams, blist, sp)
    X = X + V * sp.dt + F * sp.dt**2 / 2
    T[0] = temperature(V)
    E[0] = tot_KE(V) + tot_PE(X, iparams, blist, sp)
    save_xyzmatrix("Dump/dump_%i.xyz" % 0, blist, X)
    tf = time.time()
    print("Step: %i | T: %.5f | Time: %.2f" % (1, T[0], tf - ti))

    # Other steps
    for i in range(1, sp.Nt):
        X, V, Npass = vel_verlet_step(X, V, iparams, blist, sp)
        E[i] = tot_KE(V) + tot_PE(X, iparams, blist, sp)
        T[i] = temperature(V)
        tf = time.time()
        if (i+1) % sp.thermo == 0:
            save_xyzmatrix("Dump/dump_%i.xyz" % (i+1), blist, X)
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
        F = force_list(X, V, iparams, blist, sp)
        V = V + F * sp.dt
        X = X + V * sp.dt
        Npass = np.sum(X - X % sp.L != 0, 1)
        X = X % sp.L
        T[i] = temperature(V)
        E[i] = tot_KE(V) + tot_PE(X, iparams, blist, sp)
        tf = time.time()
        if (i+1) % sp.thermo == 0:
            save_xyzmatrix("Dump/dump_%3i.xyz" % (i+1), blist, X)
            print("Step: %i | T: %.5f | Time: %.2f" % (i+1, T[i], tf - ti))
    return T, E


# ===== code with lookup tables
def build_lookup_table(X, L, cutoff=2.0, page=150):  # fix page
    """Create a dict for each bead storing positions of 
    neighbouring beads within given cutoff"""
    N = len(X)
    lt = dict()
#    lt = -1 * np.ones(N, page)
    for i in range(N): lt[i] = []
    cell = L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        for j in range(i):
            dr = X[i] - X[j]
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            if norm(dr_n) < cutoff:
                lt[i].append(j)
                lt[j].append(i)
    return lt


def force_list_lookup(X, V, lt, iparams, blist, sp):
    """Get force matrix from lookup table"""
    N = len(X)
    force_mat = np.zeros((N, N, 3))
    cell = sp.L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        dr = X[i] - X[j]       # rij = ri - rj
        G = np.dot(inv_cell, dr)
        G_n = G - np.round(G)
        dr_n = np.dot(cell, G_n)
        v_ij = V[i] - V[j]     # vij = vi - vj
        force_mat[i, j] = F_tot(dr_n, v_ij, iparams[(blist[i], blist[j])], sp)

    force_mat -= np.transpose(force_mat, (1, 0, 2))
    return np.sum(force_mat, axis=1)
    

