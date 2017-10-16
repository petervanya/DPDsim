#!/usr/bin/env python3
"""Usage:
    main.py <input>

Simulate DPD binary mixture.
* Read input.yaml file
* Evolve for the given number of timesteps
* Print data into file and on data screen

23/06/16
"""
import numpy as np
import yaml
import os
import time
import copy
from docopt import docopt


class mydict(dict):
    """A container for all the system constants"""
    def __getattr__(self, key):
        return self[key]


if __name__ == "__main__":
    args = docopt(__doc__)
    data = yaml.load(open(args["<input>"]).read())
    seed = 1234
    np.random.seed(seed)
    
    # bead types
    bead_types = data["bead-types"]
    N = sum(bead_types.values())
    Nbt = len(bead_types.keys())
    rho = float(N/float(data["L"])**3)
    bt2num = {}
    for i, bt in enumerate(bead_types): bt2num[bt] = i
    bead_list = []
    for k, v in bead_types.items():
        bead_list += [bt2num[k]]*v

    # interaction parameters
    int_params = np.ones((Nbt, Nbt))*25.0
    for k, v in data["inter-params"].items():
        b1, b2 = k.split()
        int_params[bt2num[b1], bt2num[b2]] = v
        int_params[bt2num[b2], bt2num[b1]] = v

    sp = mydict(L=data["L"], dt=data["dt"], Nt=data["num-steps"],\
                kT=data["kT"], gamma=data["gamma"], rc=data["rc"],\
                Nbt=Nbt, thermo=data["thermo"], seed=seed, \
                saveE=data["save-energy"], use_numba=data["use-numba"],\
                method=data["integrate"])

    print(" ============== \n DPD simulation \n ==============")
    print("Beads: %i | rho: %.2f | kT: %.1f | Steps: %i | dt: %.4f | thermo: %i"
          % (N, rho, sp.kT, sp.Nt, sp.dt, sp.thermo))

    dumpdir = "Dump"
    if not os.path.exists(dumpdir):
        os.makedirs(dumpdir)

    if sp.use_numba == True:
        print("Using Numba.")
        # Is this kosher?
        from dpd_functions_numba import init_pos, init_vel, temperature, \
                integrate_verlet, integrate_euler
        print("Initialising the system...")
        X = init_pos(N, sp.L, sp.seed)
        V = init_vel(N, sp.kT)
        print("Temperature: %.2f" % temperature(V))
 
        print("%s Method. Starting integration..." % sp.method)
        ti = time.time()
        if sp.method == "Euler":
            T, E = integrate_euler(X, V, int_params, bead_list, sp)
        if sp.method == "Verlet":
            T, E = integrate_verlet(X, V, int_params, bead_list, sp)
        
    else:
        from dpd_functions import init_pos, init_vel, temperature, \
                integrate_verlet, integrate_euler
        print("Initialising the system...")
        X = init_pos(N, sp.L, sp.seed)
        V = init_vel(N, sp.kT)
        print("Temperature: %.2f" % temperature(V))
 
        print("%s Method. Starting integration..." % sp.method)
        ti = time.time()
        if sp.method == "Euler":
            T, E = integrate_euler(X, V, int_params, bead_list, sp)
        if sp.method == "Verlet":
            T, E = integrate_verlet(X, V, int_params, bead_list, sp)
    tf = time.time()
    print("Simulation time: %.2f s." % (tf - ti))
    print("Relative energy difference: %.3e." % ((E[-1] - E[0]) / E[0]))


