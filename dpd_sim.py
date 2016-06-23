#!/usr/bin/env python
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
from docopt import docopt
from dpd_functions import init_pos, init_vel


class mydict(dict):
    """A container for all the system constants"""
    def __getattr__(self, key):
        return self[key]


if __name__ == "__main__":
    args = docopt(__doc__)
    data = yaml.load(open(args["<input>"]).read())
    seed = 1234
    np.random.seed(seed)
    
    sp = mydict(L=data["L"], dt=data["dt"], numsteps=data["num-steps"],\
                kBT=data["kBT"], gamma=data["gamma"], rc=data["rc"],\
                thermo=data["thermo"], seed=seed)

    # bead types
    bead_types = data["bead-types"]
    N = sum(bead_types.values())
    Nbt = len(bead_types.keys())
    bt2num = {}
    for i, bt in enumerate(bead_types): bt2num[bt] = i+1
    bead_list = []
    for k, v in bead_types.items():
        bead_list += [bt2num[k]]*v

    # interaction parameters
    inter_params = {}
    for k, v in data["inter-params"].items():
        b1, b2 = k.split()
        inter_params[(bt2num[b1], bt2num[b2])] = v
        inter_params[(bt2num[b2], bt2num[b1])] = v
    print(inter_params)

    print(" ============== \n DPD simulation \n ==============")
    print("Particles: %i | Temp: %.1f | Steps: %i | dt: %.2f | thermo: %i"
          % (N, data["kBT"], data["num-steps"], data["dt"], data["thermo"]))

    dumpdir = "Dump"
    if not os.path.exists(dumpdir):
        os.makedirs(dumpdir)

    # init system
    print("Initialising the system...")
    pos_list, E = init_pos(N, inter_params, sp)
    vel_list = init_vel(N, data["kBT"])
#
#    # run system
#    print("Starting integration...")
#    xyz_frames, E = integrate(pos_list, vel_list, sp)
#
#    # print into file
#    Nf = xyz_frames.shape[-1]
#        for i in range(Nf):
#            fname = "Dump/dump_" + str((i+1)*thermo) + ".xyz"
#            save_xyzmatrix(fname, xyz_frames[:, :, i])
#    print_timing()
