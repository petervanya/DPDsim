#!/usr/bin/env python3
"""
Input-output routines

"""
import numpy as np
import sys


def save_xyzfile(fname, names, mat):
    """Save a coordinate matrix in a xyz format"""
    M = len(mat)
    s = "%i\nbla\n" % M
    for i in range(M):
        s += "%i\t%.10f\t%.10f\t%.10f\n" % \
                (names[i], mat[i, 0], mat[i, 1], mat[i, 2])
    open(fname, "w").write(s)


def read_xyzfile(fname):
    try:
        A = open(fname, "r").readlines()[2:]
    except FileNotFoundError:
        sys.exit("File %s not found." % fname)
    A = [line.split() for line in A]
    A = np.array(A).astype(float)
    if A.shape[1] != 4:
        sys.exit("Incorrect number of columns in %s." % fname)
    nm, xyz = A[:, 0], A[:, 1:]
    return nm, xyz


def parse_box(arg):
    L = [float(i) for i in arg.split()]
    if len(L) == 1:
        box = np.eye(3) * L[0]
    elif len(L) == 3:
        box = np.diag(L)
    else:
        sys.exit("L should have length 1 or 3.")
    return box


