# DPDsim

[![DOI:10.1103/PhysRevE.102.013312](https://img.shields.io/badge/doi-10.1103%2FPhysRevE.102.013312-blue.svg)](https://doi.org/10.1103/PhysRevE.102.013312)

An implementation of several variants of dissipative particle dynamics, a coarse-grained molecular dynamics method.

Available methods:
1. Standard dissipative particle dynamics (DPD), source: [Groot and Warren, JCP, 1997](https://doi.org/10.1063/1.474784)
2. Many-body dissipative particle dynamics (MDPD), source: [Warren, PRE, 2003](https://doi.org/10.1103/PhysRevE.68.066702)
3. A generalised many-body DPD with freedom in defining local density and wrapping function (GMDPD)
   presented in [Vanya and Elliott, PRE, 2020](https://doi.org/10.1103/PhysRevE.102.013312)

This implementation of DPD/MDPD allows non-bonded particles only. 
For more complex simulations of polymers, we recommend [DL_MESO](https://www.scd.stfc.ac.uk/Pages/DL_MESO.aspx) 
or [LAMMPS](https://github.com/lammps/lammps).


## Code versions
Project is mainly written in Python.
Key bottlenecks involving force and energy computation are optimised as follows:
* Enhancement by Numba library (experimental)
* A Fortran module linked via f2py (about 10x faster)

Post-processing tools to compute density profiles and radial distribution functions are in `utils.py`.

NB: There are also ad hoc versions in pure Fortran and Julia not developed anymore.


## Requirements
* Python3
* numpy, numba, pandas (use `requirements.txt`)
* a Fortran compiler (gnu95 is used here)


## Example
Pure DPD with Fortran-optimised code:
```Python
from dpdsim import DPDSim
sim = DPDSim(implementation="fortran", steps=1000, thermo=100)
sim.create_particle_inputs(kind="pure") # alternative is 'binary'
sim.run()
```

Other force fields (MDPD, EMDPD, GMDPD) are handled in the same way.

Access to key functions:
```Python
sim.compute_pe()             # returns potential enerigy
sim.compute_ke()             # returns kinetic energy
sim.compute_local_density()  # computes MDPD local density stored in "sim.rho2"
sim.compute_force()          # computes force, virial and the stress tensor diagonal
```

Post-processing from within Python:
```Python
from dpdsim.utils import compute_profile, compute_rdf
# run with DPDSim object "sim"
r_pr, pr = compute_profile(sim, 0, 1, N_bins=50) # 0 for x-coord, 1 for particle type
r_rdf, rdf = compute_rdf(sim, 1, N_bins=50)      # 1 for particle type
```
