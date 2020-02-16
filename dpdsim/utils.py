#!/usr/bin/env python
"""
Collection of various utility functions.
"""
import numpy as np
from numba import jit
import time


def compute_profile(sim, ax, bt, nbins=100):
    """
    Compute a profile along the specificed coordinate.
    """
    L = np.diag(sim.box)
    bins = np.linspace(0, L[ax], nbins+1)
    dr = bins[1] - bins[0]
    r = dr / 2 + bins[:-1]

    Lsurf = L[list(set(range(3)).difference([ax]))] # cross-sectional surface

    pr = np.zeros(len(bins)-1)

    if bt == -1:    # choose all beads
        atoms = sim.X
    else:
        atoms = sim.X[sim.bl == bt]
    pr, _ = np.histogram(atoms[:, ax], bins=bins)

    pr = pr / (dr * np.prod(Lsurf))
    return r, pr


def compute_rdf(sim, bt, N_bins=100, verbose=False):
    if type(bt) == int:
        bts = [bt]
    elif type(bt) in [list, tuple]:
        bts = list(bt)
    else:
        raise TypeError("Wrong type of bt: %s." % type(bt))
    
    bins = np.linspace(0, np.min(np.diag(sim.box)) / 2, N_bins + 1)
    dr = bins[1] - bins[0]
    r = dr / 2.0 + bins[:-1]

    if not set(bts).issubset(sim.bl):
        raise ValueError("Requested bead type not present.")

    if verbose:
        print("Calculating rdf...")

    ti = time.time()
    if len(bts) == 1:
        X = sim.X[sim.bl == bts[0]]
        rdf = compute_rdf_1_type(X, r, dr, sim.box)
    elif len(bts) == 2:
        X1 = sim.X[sim.bl == bts[0]]
        X1 = sim.X[sim.bl == bts[1]]
        rdf = compute_rdf_2_types(X1, X2, r, dr, sim.box)
    tf = time.time()
    print("Time: %.2f s" % (tf - ti))

    return r, rdf


@jit#(nopython=True)
def compute_rdf_1_type(X, r, dr, box):
    rdf = np.zeros(r.shape)
    hist = np.zeros(r.shape, dtype=int)
    N = len(X)
    dX = np.zeros(3)
    g = np.zeros(3)
    a = 0.0

    inv_box = np.linalg.pinv(box)
    volume = np.prod(np.diag(box))
    Npair = N * (N-1) / 2.0

    for i in range(N):
        for j in range(i):
            dX = X[i] - X[j]
            g = inv_box.dot(dX)
            g = g - np.round_(g) # FIX for Numba
            a = np.sqrt(((box.dot(g))**2).sum())

            pos = int(a//dr)
            if pos < len(r):
                hist[pos] += 1

    rdf = hist / (4*np.pi*r**2 * dr) * volume / Npair
    return rdf


@jit(nopython=True)
def compute_rdf_2_types(X1, X2, r, dr, box):
    raise NotImplementedError()







