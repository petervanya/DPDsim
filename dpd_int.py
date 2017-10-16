#/usr/bin/env python
"""
Module storing integrating functions.

16/10/17
"""
import numpy as np
from dpd_io import save_xyzfile


def integrate_f(X, V, bl, ip, box, gama, kT, dt, \
        Nt, Neq, thermo, df, style):
    """Integrate using dpd_f.f90 module"""
    from Fdpd.dpd_f import dpd_f
    N = len(X)
    KE = np.zeros(Nt+1)
    PE = np.zeros(Nt+1)
    T = np.zeros(Nt+1)

    F = dpd_f.force_mat(X, V, bl, ip, box, gama, kT, dt)
    for it in range(1, Nt+1):
        if style == "euler":
            _ = dpd_f.euler_step(X, V, bl, ip, box, gama, kT, dt)
        elif style == "verlet":
            _ = dpd_f.verlet_step(X, V, F, bl, ip, box, gama, kT, dt)

        X = X % np.diag(box)
        KE[it] = dpd_f.tot_ke(V)
        PE[it] = dpd_f.tot_pe(X, bl, ip, box)
        T[it] = KE[it] / ((3 * N - 3) / 2.0)
        if it % thermo == 0:
            print("Step: %3.i | T: %.5f | KE: %.3e | PE: %.3e" % \
                (it, T[it], KE[it], PE[it]))
        if it >= Neq and it % df == 0:
            save_xyzfile("Dump/dump_%05i.xyz" % it, bl, X)
    return T, KE, PE


def integrate_numba(X, V, bl, ip, box, gamma, kT, dt,\
        Nt, Neq, thermo, df, style):
    from dpd_functions import euler_step, verlet_step, \
            tot_KE, tot_PE, force_mat
    N = len(X)
    T = np.zeros(Nt+1)
    KE = np.zeros(Nt+1)
    PE = np.zeros(Nt+1)

    F = force_mat(X, V, bl, ip, box, gamma, kT, dt)
    for it in range(1, Nt+1):
        if style == "euler":
            X, V = euler_step(X, V, bl, ip, box, gamma, kT, dt)
        elif style == "verlet":
            X, V, F = verlet_step(X, V, F, bl, ip, box, gamma, kT, dt)

        X = X % np.diag(box)
        KE[it] = tot_KE(V)
        PE[it] = tot_PE(X, bl, ip, box)
        T[it] = KE[it] / ((3 * N - 3) / 2.0)
        if it % thermo == 0:
            print("Step: %3.i | T: %.5f | KE: %.5e | PE: %.5e" % \
                (it, T[it], KE[it], PE[it]))
        if it >= Neq and it % df == 0:
            save_xyzfile("Dump/dump_%05i.xyz" % it, bl, X)
    return T, KE, PE


