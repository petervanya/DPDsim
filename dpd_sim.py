#!/usr/bin/env python3
"""Usage:
    dpd_sim.py <method> [--L <L> --N <N> --dt <dt> --gamma <g>]
               [--read <xyz> --steps <ns> --eq <eq> --thermo <th>]
               [--dump-freq <df> --seed <s> --style <st>]

Simulate a DPD binary mixture.

Options:
    <method>           "numba" or "fortran"

Options:
    --N <N>            Number of atoms [default: 375]
    --L <L>            Box size [default: 5]
    --dt <dt>          Timestep [default: 0.05]
    --steps <ns>       Number of steps [default: 100]
    --eq <eq>          Equilibration steps [default: 90]
    --thermo <th>      Frequency of printing to screen [default: 10]
    --dump-freq <df>   Frequency of printing to file [default: 10]
    --gamma <g>        Friction [default: 4.5]
    --read <xyz>       Read the xyz file, else generate beads randomly
    --seed <s>         Random seed [default: 1234]
    --style <st>       Integration style [default: euler]

14/10/17
"""
import numpy as np
import os, time
from docopt import docopt
from dpd_io import read_xyzfile, parse_box


if __name__ == "__main__":
    args = docopt(__doc__)
    np.random.seed(int(args["--seed"]))
    N = int(args["--N"])
    box = parse_box(args["--L"])
    rho = float(N / np.prod(np.diag(box)))
    gamma = float(args["--gamma"])
    kT = 1.0
    dt = float(args["--dt"])
    Nt = int(args["--steps"])
    Neq = int(args["--eq"])
    thermo = int(args["--thermo"])
    df = int(args["--dump-freq"])
    style = args["--style"]
    styles = ["euler", "verlet"]
    if style.lower() not in styles:
        sys.exit("Choose style from %s." % styles)

    dumpdir = "Dump"
    if not os.path.exists(dumpdir): os.makedirs(dumpdir)

    if args["--read"]:
        bl, X = read_xyzfile(args["--read"])
        X = X % np.diag(box)
        Nbt = len(set(bl))
        ip = np.zeros((Nbt, Nbt))
        ip = 25.0
    else:
        X = np.random.rand(N, 3) * np.diag(box)
        bl = np.ones(N).astype(int)
        ip = np.array([[25.0]])

    V = np.random.randn(N, 3) * kT
    V = V - np.sum(V, 0) / N

    print("===== DPD simulation =====")
    print("N: %i | L: %s | rho: %.2f" % (N, np.diag(box), rho))
    print("gamma: %.2f | kT: %.1f | dt: %.3f" % (gamma, kT, dt))
    print("Steps: %i | eq: %i | Thermo: %i | dump-freq: %i" \
            % (Nt, Neq, thermo, df))

    ti = time.time()
    mt = args["<method>"].lower()
    if mt == "numba":
        from dpd_int import integrate_numba
        T, KE, PE = integrate_numba(X, V, bl, ip, box, gamma, kT, dt, \
                Nt, Neq, thermo, df, style=style)
    elif mt == "fortran":
        from dpd_int import integrate_f
        X = np.asfortranarray(X)
        V = np.asfortranarray(V)
        T, KE, PE = integrate_f(X, V, bl, ip, box, gamma, kT, dt, \
                Nt, Neq, thermo, df, style=style)

    print("Simulation time: %.2f s." % (time.time() - ti))


