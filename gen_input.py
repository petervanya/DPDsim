#!/usr/bin/env python
"""Usage:
    gen_input.py [--L <L> --dt <dt> --steps <N> --kT <kT> --thermo <th>]
                 [--NA <NA> --NB <NB> --aii <aii> --aij <aij>]
                 [--int <int>]

Generate input for the simulation.

Options:
    --L <L>        Box size [default: 5]
    --dt <dt>      Time step [default: 0.02]
    --steps <N>    Number of time steps [default: 100]
    --kT <kT>      Temperature in DPD units [default: 1.0]
    --NA <NA>      Number of A beads [default: 200]
    --NB <NB>      Number of B beads [default: 200]
    --aii <aii>    Same bead interaction [default: 25]
    --aij <aij>    Different beads interaction [default: 40]
    --thermo <th>  How often write into file [default: 1]
    --int <int>    Choose method, Euler or Verlet [default: Euler]

24/06/16
"""
from docopt import docopt

args = docopt(__doc__)
L = float(args["--L"])
dt = float(args["--dt"])
steps = int(args["--steps"])
kT = float(args["--kT"])
gamma = 4.5
rc = 1.0
thermo = int(args["--thermo"])
NA = int(args["--NA"])
NB = int(args["--NB"])
aii = float(args["--aii"])
aij = float(args["--aij"])
saveE = False
use_numba = False
method = args["--int"]


s = "# Input file for home-made DPD simulation\n"
s += "L: %.1f\n" % L
s += "dt: %.2f\n" % dt
s += "num-steps: %i\n" % steps
s += "kT: %.1f\n" % kT
s += "gamma: %.1f\n" % gamma
s += "rc: %.1f\n" % rc
s += "thermo: %i        # print coords this many times\n" % thermo

s += "bead-types:     # number (future add: mass)\n"
s += "    A: %i\n" % NA
s += "    B: %i\n" % NB

s += "inter-params:   # a_ij (future add: gamma, rc)\n"
s += "    A A: %.1f\n" % aii
s += "    A B: %.1f\n" % aij
s += "    B B: %.1f\n" % aii
s += "save-energy: %r\n" % saveE
s += "integrate:   %s\n" % method
s += "use-numba:   %r\n" % use_numba

open("input.yaml", "w").write(s)
print("File written in input.yaml")


