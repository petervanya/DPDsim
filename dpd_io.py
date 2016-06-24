#!/usr/bin/env python3
"""
Input-output routines

"""
import numpy as np


def save_xyzmatrix(fname, names, mat):
    """Save a coordinate matrix in a xyz format compatible with VMD"""
    M = mat.shape[0]
    s = "%i\nbla\n" % M
    for i in range(M):
        s += "%i\t%.3f\t%.3f\t%.3f\n" % (names[i], mat[i, 0], mat[i, 1], mat[i, 2])
    open(fname, "w").write(s)


def read_xyzmatrix(fname):
    A = open(fname, "r").readlines()[2:]
    A = [line.split() for line in A]
    A = np.array(A).astype(float)
    return A
