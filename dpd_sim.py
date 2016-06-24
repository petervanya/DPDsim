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
from docopt import docopt
from dpd_functions import init_pos, init_vel, integrate, temperature


class mydict(dict):
    """A container for all the system constants"""
    def __getattr__(self, key):
        return self[key]


if __name__ == "__main__":
    args = docopt(__doc__)
    data = yaml.load(open(args["<input>"]).read())
    seed = 1234
    np.random.seed(seed)
    
    sp = mydict(L=data["L"], dt=data["dt"], Nt=data["num-steps"],\
                kBT=data["kBT"], gamma=data["gamma"], rc=data["rc"],\
                thermo=data["thermo"], seed=seed, saveE=data["save-energy"])

    # bead types
    bead_types = data["bead-types"]
    N = sum(bead_types.values())
    rho = float(N/sp.L**3)
    Nbt = len(bead_types.keys())
    bt2num = {}
    for i, bt in enumerate(bead_types): bt2num[bt] = i+1
    bead_list = []
    for k, v in bead_types.items():
        bead_list += [bt2num[k]]*v

    # interaction parameters
    int_params = {}
    for k, v in data["inter-params"].items():
        b1, b2 = k.split()
        int_params[(bt2num[b1], bt2num[b2])] = v
        int_params[(bt2num[b2], bt2num[b1])] = v

    print(" ============== \n DPD simulation \n ==============")
    print("Beads: %i | rho: %.2f | kBT: %.1f | Steps: %i | dt: %.2f | thermo: %i"
          % (N, rho, sp.kBT, sp.Nt, sp.dt, sp.thermo))

    dumpdir = "Dump"
    if not os.path.exists(dumpdir):
        os.makedirs(dumpdir)

    # init system
    print("Initialising the system...")
    pos_list, E = init_pos(N, int_params, bead_list, sp)
    vel_list = init_vel(N, sp.kBT)
    print("Temperature: %.2f" % temperature(vel_list))

    # run system
    print("Starting integration...")
    ti = time.time()
    T, E = integrate(pos_list, vel_list, int_params, bead_list, sp)
    tf = time.time()
    print("Simulation time: %.2f s." % (tf-ti))


