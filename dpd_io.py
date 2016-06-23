#!/usr/bin/env python3
"""
Input-output routines

"""
import numpy as np


def save_xyzmatrix(fname, mat, init_char="1"):
    """Save a coordinate matrix in a xyz format compatible with VMD"""
    M, N = mat.shape
    with open(fname, "w") as f:
        f.write(str(M) + "\nbla\n")
        for i in range(M):
            line = init_char + "   "
            for j in range(N):
                line += str(mat[i, j]) + "   "
            line += "\n"
            f.write(line)


def read_xyzmatrix(fname):
    A = open(fname, "r").readlines()[2:]
    A = [line.split() for line in A]
    A = np.array(A).astype(float)
    return A
