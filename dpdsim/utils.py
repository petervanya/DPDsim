#!/usr/bin/env python
"""
Collection of various utility functions.
"""
import numpy as np
from numba import jit
import time
import glob


def compute_profile(sim, ax, bt, N_bins=100, shift=None):
    """
    Compute a profile along the specificed coordinate.

    Input
    =====
    - ax: axis, from (0, 1, 2)
    - bt: bead type, from available types, -1 for all
    - N_bins: number of histogram bins
    """
    L = np.diag(sim.box)
    bins = np.linspace(0, L[ax], N_bins+1)
    dr = bins[1] - bins[0]
    r = dr / 2 + bins[:-1]

    Lsurf = L[list(set(range(3)).difference([ax]))] # cross-sectional surface
    pr = np.zeros_like(r)

    X = sim.X.copy()
    if shift is not None:
        assert hasattr(shift, "__iter__") and len(shift) == 3, \
            "Vector of shifting must be of size 3."
        shift = np.array(shift)
        X = X + shift
        X = X % L

#    if bt == -1:    # choose all beads
#        X = sim.X
    if bt != -1:
        X = X[sim.bl == bt]
    pr, _ = np.histogram(X[:, ax], bins=bins)

    pr = pr / (dr * np.prod(Lsurf))
    return r, pr


def compute_profile_from_frames(frames_str, ax, bt, box, N_bins=100, \
    shift=None, verbose=False):
    """
    Compute a density profile from a batch of xyz frames.
    
    Input
    =====
    - frames_str: a regex containing frames in xyz format
    - ax: axis along which to compute the profile
    - bt: bead type
    - box: box size, a (3, 3) matrix
    - N_bins: number of bins

    Output
    ======
    - r: position vector
    - pr: density profile vector
    """
    frames = glob.glob(frames_str)
    assert len(frames) != 0, "No xyz frames captured."
    Nf = len(frames)
    N = int(open(frames[0], "r").readline())
    if verbose:
        print(frames)
    
    L = np.diag(box)
    bins = np.linspace(0, L[ax], N_bins + 1)
    dr = bins[1] - bins[0]
    r = dr / 2.0 + bins[:-1]

    Lsurf = L[list(set(range(3)).difference([ax]))] # cross-sectional surface
    pr = np.zeros_like(r)

    for frame in frames:
        bl, X0 = read_xyz(frame)

        if shift is not None:
            assert len(shift) == 3, "Vector of shifting must be of size 3."
            shift = np.array(shift)
            X0 = X0 + shift
            X0 = X0 % L

        if bt == -1:
            X = X0
        else:
            X = X0[bl == bt]
        pr += np.histogram(X[:, ax], bins=bins)[0]

    pr = pr / (dr * np.prod(Lsurf)) / Nf
    return r, pr


def compute_rdf(sim, bt, N_bins=100, verbose=False):
    """
    Compute rdf for a 

    Input
    =====
    - sim: a dpdsim object containing coordinates and box
    - bt: bead type

    Output
    ======
    - r: a vector of positions
    - rdf: a vector of d radial distribution function
    - N_bins: number of bins [default: 100]
    """
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

    tic = time.time()
    if len(bts) == 1:
        X = sim.X[sim.bl == bts[0]]
        rdf = compute_rdf_1_type(X, r, dr, sim.box)
    elif len(bts) == 2:
        X1 = sim.X[sim.bl == bts[0]]
        X2 = sim.X[sim.bl == bts[1]]
        rdf = compute_rdf_2_types(X1, X2, r, dr, sim.box)
    toc = time.time()
    print("Time: %.2f s" % (toc - tic))

    return r, rdf


def compute_rdf_from_frames(frames_str, bt, box, N_bins=100, verbose=False):
    """
    Read in a batch of xyz frames via regex and compute
    a radial distribution function.
    
    Input
    =====
    - frames_str: a regex containing frames in xyz format
    - bt: bead type
    - box: box size, a (3, 3) matrix
    - N_bins: number of bins [default: 100]

    Output
    ======
    - r: a vector of positions
    - rdf: a vector of d radial distribution function
    """
    frames = glob.glob(frames_str)
    assert len(frames) != 0, "No xyz frames captured."
    Nf = len(frames)
    N = int(open(frames[0], "r").readline())
    if verbose:
        print(frames)
    
    if type(bt) == int:
        bts = [bt]
    elif type(bt) in [list, tuple]:
        bts = list(bt)
    else:
        raise TypeError("Wrong type of bt: %s." % type(bt))
    
    if len(np.array(box).shape) == 1:
        box = np.diag(box) # convert to numpy matrix
    box = np.array(box).astype(float) # ensure correct data type
    bins = np.linspace(0, np.min(np.diag(box)) / 2, N_bins + 1)
    dr = bins[1] - bins[0]
    r = dr / 2.0 + bins[:-1]
    rdf = np.zeros_like(r)

    bl = set(read_xyz(frames[0])[0])
    if not set(bts).issubset(bl):
        raise ValueError("Requested bead type not present.")
    if len(bts) > 2:
        raise ValueError("At most two bead types allowed.")

    if verbose:
        print("Calculating rdf...")

    tic = time.time()
    for frame in frames:
        bl, X0 = read_xyz(frame)
        if len(bts) == 1:
            X = X0[bl == bts[0]]
            rdf += compute_rdf_1_type(X, r, dr, box)
        elif len(bts) == 2:
            X1 = X0[bl == bts[0]]
            X2 = X0[bl == bts[1]]
            rdf += compute_rdf_2_types(X1, X2, r, dr, box)
    toc = time.time()
    print("Time: %.2f s" % (toc - tic))

    rdf /= Nf
    return r, rdf


@jit(nopython=True)
def compute_rdf_1_type(X, r, dr, box):
    N = len(X)
    rdf = np.zeros(r.shape)
    hist = np.zeros(r.shape) #, dtype=int) # does not work with Numba!
    dX = np.zeros(3)
    g = np.zeros(3)
    d = 0.0

    inv_box = np.linalg.pinv(box)
    volume = np.prod(np.diag(box))
    Npair = N * (N-1) / 2.0

    for i in range(N):
        for j in range(i):
            dX = X[i] - X[j]
            g = inv_box.dot(dX)
            g = g - round_numba(g)
            d = np.sqrt(((box.dot(g))**2).sum())

            pos = int(d//dr)
            if pos < len(r):
                hist[pos] += 1

    rdf = hist / (4*np.pi*r**2 * dr) * volume / Npair
    return rdf


@jit(nopython=True)
def compute_rdf_2_types(X1, X2, r, dr, box):
    N1 = len(X1)
    N2 = len(X2)
    rdf = np.zeros(r.shape)
    hist = np.zeros(r.shape)
    dX = np.zeros(3)
    g = np.zeros(3)
    d = 0.0

    inv_box = np.linalg.pinv(box)
    volume = np.prod(np.diag(box))
    Npair = N1 * N2

    for i in range(N1):
        for j in range(N2):
            dX = X1[i] - X2[j]
            g = inv_box.dot(dX)
            g = g - round_numba(g)
            d = np.sqrt(((box.dot(g))**2).sum())

            pos = int(d//dr)
            if pos < len(r):
                hist[pos] += 1

    rdf = hist / (4*np.pi*r**2 * dr) * volume / Npair
    return rdf



@jit(nopython=True)
def round_numba(g):
    """Does not work with nopython"""
    N = len(g)
    gr = np.zeros(N)
    for i in range(N):
        gr[i] = round(g[i])
    return gr


def read_xyz(outfile):
    """Read one xyz outfile into a numpy matrix.
    Return vector of names and (n, 3) xyz matrix."""
    try:
        A = open(outfile, "r").readlines()[2:]
    except FileNotFoundError:
        print("File %s not found." % outfile)
        raise

    A = open(outfile, "r").readlines()[2:]
    A = np.array([line.split() for line in A]).astype(float)
    bl, xyz = A[:, 0].astype(int), A[:, 1:4]
    return bl, xyz






