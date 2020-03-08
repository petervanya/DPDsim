# DPDsim

An implementation of several variants of dissipative particle dynamics,
a coarse-grained molecular dynamics method.

1. Standard dissipative particle dynamics (DPD), source: [Groot and Warren, JCP, 1997](https://doi.org/10.1063/1.474784)
2. Many-body dissipative particle dynamics (MDPD), source: [Warren, PRE, 2003](https://doi.org/10.1103/PhysRevE.68.066702)
3. A generalised many-body DPD with freedom in defining local density and wrapping function (GMDPD)

This implementation of DPD/MDPD allows non-bonded particles only. 
For more complex simulations of polymers, [DL_MESO](https://www.scd.stfc.ac.uk/Pages/DL_MESO.aspx) 
or [LAMMPS](https://github.com/lammps/lammps) are recommended.


## Code versions
The default language is Python. Key bottlenecks involving force and energy computation
are dealt with in two ways:
* A Fortran module linked via f2py
* Enhancement with Numba (a JIT compiler)

There are ad hoc versions in pure Fortran and Julia, which are not developed anymore.

The recommended and most developed way to simulate is via the package `dpdsim`.

Post-processing tools to compute density profiles and radial distribution functions
are `utils.py`.


## Example
Pure DPD with Fortran-optimised code:
```Python
from dpdsim import DPDSim
sim = DPDSim(implementation="fortran", steps=1000, thermo=100)
sim.create_particle_inputs(kind="pure") # alternative is binary
sim.run()
```

Other force fields are handled in the same way.

Separate functions can be accessed as follows:
```Python
sim.compute_pe() # returns potential enerigy
sim.compute_ke() # returns kinetic energy
sim.compute_local_density() # computes MDPD local density
sim.compute_force() # computes force, virial and the stress tensor diagonal
```

## Requirements
* a Fortran compiler
* numpy, numba, pandas, docopt


## Performance
Fortran-optimised code is about an order of magnitude faster than Python/Numba.




