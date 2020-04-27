#/usr/bin/env python
"""
Module storing integrating functions.

16/10/17
"""
import numpy as np
from .sim_io import save_xyzfile


def integrate_numba(X, V, bl, ip, box, gamma, kT, dt,\
        Nt, Neq, thermo, df, dump_vel, dump_for, style):
    from .forces import euler_step, verlet_step, \
            compute_ke, compute_pe, compute_force
    N = len(X)
    volume = np.linalg.det(box)
    rho = N / volume

    T = np.zeros(Nt+1)
    KE = np.zeros(Nt+1)
    PE = np.zeros(Nt+1)
    P = np.zeros(Nt+1)
    Pxx, Pyy, Pzz = np.zeros(Nt+1), np.zeros(Nt+1), np.zeros(Nt+1)

    F, vir, sigma = compute_force(X, V, bl, ip, box, gamma, kT, dt)

    # save first screenshot
    save_xyzfile("Dump/dump_%05i.xyz" % 0, bl, X)
    if dump_vel:
        save_xyzfile("Dump/dump_%05i.vel" % 0, bl, V)
    if dump_for:
        save_xyzfile("Dump/dump_%05i.for" % 0, bl, F)

    print("step temp ke pe p pxx pyy pzz")
    for it in range(1, Nt+1):
        if style == "euler":
            X, V, _, vir, sigma = euler_step(X, V, bl, ip, box, gamma, kT, dt)
        elif style == "verlet":
            X, V, F, vir, sigma = verlet_step(X, V, F, bl, ip, box, gamma, kT, dt)

        X = X % np.diag(box)

        ke = compute_ke(V)
        pe = compute_pe(X, bl, ip, box)
        temp = ke / ((3*N - 3) / 2.0)
        pxx, pyy, pzz = sigma #np.diag(sigma)
        p = (pxx + pyy + pzz) / 3.0
#        p = rho * kT + vir / (3 * volume)

        KE[it], PE[it], T[it], P[it], Pxx[it], Pyy[it], Pzz[it] = \
            ke, pe, temp, p, pxx, pyy, pzz

        if it % thermo == 0:
            print("%3.i %.5f %.3e %.3e %.3e %.3e %.3e %.3e" % \
                (it, temp, ke, pe, p, pxx, pyy, pzz))

        if it >= Neq and it % df == 0:
            save_xyzfile("Dump/dump_%05i.xyz" % it, bl, X)
            if dump_vel:
                save_xyzfile("Dump_vel/dump_%05i.vel" % it, bl, V)
            if dump_for:
                save_xyzfile("Dump_for/dump_%05i.for" % it, bl, F)

    return T, KE, PE, P, Pxx, Pyy, Pzz


# =====
# Fortran integration functions
# =====
def integrate_f(X, V, bl, ip, box, gama, kT, dt, \
        Nt, Neq, thermo, df, dump_vel, dump_for, style):
    """Integrate using dpd_f.f90 module"""
    from Fdpd.dpd_f import dpd_f
    N = len(X)

    KE = np.zeros(Nt+1)
    PE = np.zeros(Nt+1)
    T = np.zeros(Nt+1)
    F = dpd_f.force_mat(X, V, bl, ip, box, gama, kT, dt)

    save_xyzfile("Dump/dump_%05i.xyz" % 0, bl, X)
    if dump_vel:
        save_xyzfile("Dump/dump_%05i.vel" % 0, bl, V)
    if dump_for:
        save_xyzfile("Dump/dump_%05i.for" % 0, bl, F)

    print("step temp ke pe")
    for it in range(1, Nt+1):
        if style == "euler":
            _ = dpd_f.euler_step(X, V, bl, ip, \
                    box, gama, kT, dt)
        elif style == "verlet":
            _ = dpd_f.verlet_step(X, V, F, bl, ip, \
                    box, gama, kT, dt)

        X = X % np.diag(box)

        KE[it] = dpd_f.tot_ke(V)
        PE[it] = dpd_f.tot_pe(X, bl, ip, box)
        T[it] = KE[it] / ((3 * N - 3) / 2.0)

        if it % thermo == 0:
            print("%3.i %.5f %.3e %.3e" % (it, T[it], KE[it], PE[it]))
        if it >= Neq and it % df == 0:
            save_xyzfile("Dump/dump_%05i.xyz" % it, bl, X)
            if dump_vel:
                save_xyzfile("Dump_vel/dump_%05i.vel" % it, bl, V)
            if dump_for:
                save_xyzfile("Dump_for/dump_%05i.for" % it, bl, F)

    return T, KE, PE


def integrate_f_mdpd(X, V, bl, Ntp, ip, box, gama, kT, dt, ipb, rd, \
        Nt, Neq, thermo, df, dump_vel, dump_for, style):
    """Integrate using dpd_f.f90 module"""
    from Fdpd.mdpd_f import mdpd_f
    N = len(X)

    KE = np.zeros(Nt+1)
    PE = np.zeros(Nt+1)
    T = np.zeros(Nt+1)
    F = mdpd_f.force_mat(X, V, bl, Ntp, ip[1:, 1:], 
            box, gama, kT, dt, ipb[1:, 1:], rd)

    save_xyzfile("Dump/dump_%05i.xyz" % 0, bl, X)
    if dump_vel:
        save_xyzfile("Dump/dump_%05i.vel" % 0, bl, V)
    if dump_for:
        save_xyzfile("Dump/dump_%05i.for" % 0, bl, F)

    print("step temp ke pe(rep term)")
    for it in range(1, Nt+1):
        if style == "euler":
            _ = mdpd_f.euler_step(X, V, bl, Ntp, ip[1:, 1:], \
                    box, gama, kT, dt, ipb[1:, 1:], rd)
        elif style == "verlet":
            _ = mdpd_f.verlet_step(X, V, F, bl, Ntp, ip[1:, 1:], \
                    box, gama, kT, dt, ipb[1:, 1:], rd)

        X = X % np.diag(box)

        KE[it] = mdpd_f.tot_ke(V)
        PE[it] = mdpd_f.tot_pe(X, bl, ip[1:, 1:], box)
        T[it] = KE[it] / ((3 * N - 3) / 2.0)

        if it % thermo == 0:
            print("%3.i %.5f %.3e %.3e" % (it, T[it], KE[it], PE[it]))
        if it >= Neq and it % df == 0:
            save_xyzfile("Dump/dump_%05i.xyz" % it, bl, X)
            if dump_vel:
                save_xyzfile("Dump_vel/dump_%05i.vel" % it, bl, V)
            if dump_for:
                save_xyzfile("Dump_for/dump_%05i.for" % it, bl, F)

    return T, KE, PE




