# DPDsim

An implementation of several variants of dissipative particle dynamics,
a coarse-grained molecular dynamics method.

1. Standard dissipative particle dynamics (DPD), source: [Groot and Warren, JCP, 1997](https://doi.org/10.1063/1.474784)
2. Many-body dissipative particle dynamics (MDPD), source: [Warren, PRE, 2003](https://doi.org/10.1103/PhysRevE.68.066702)
3. A generalised many-body DPD with freedom in defining local density and wrapping function (GMDPD)
   presented in [Vanya and Elliott, PRE, 2020](https://doi.org/10.1103/PhysRevE.102.013312)

This implementation of DPD/MDPD allows non-bonded particles only. 
For more complex simulations of polymers, [DL_MESO](https://www.scd.stfc.ac.uk/Pages/DL_MESO.aspx) 
or [LAMMPS](https://github.com/lammps/lammps) are recommended.


## Code versions
The default language is Python. 
Key bottlenecks involving force and energy computation are optimised in two ways:
* A Fortran module linked via f2py
* Enhancement with Numba (a JIT compiler)

Post-processing tools to compute density profiles and radial distribution functions
are in `utils.py`.

There are ad hoc versions in pure Fortran and Julia not developed anymore.


## Requirements
* Python 3
* numpy, numba, pandas
* a Fortran compiler (gnu95 is used here)


## Example
Pure DPD with Fortran-optimised code:
```Python
from dpdsim import DPDSim
sim = DPDSim(implementation="fortran", steps=1000, thermo=100)
sim.create_particle_inputs(kind="pure") # alternative is binary
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


## Performance
Fortran-optimised code is about an order of magnitude faster than Python/Numba.




