# DPDsim
23/06/16

An implementation of several variants of dissipative particle dynamics,
a coarse-grained molecular dynamics method.

* Standard dissipative particle dynamics (DPD), source: [Groot and Warren, JCP, 1997](https://doi.org/10.1063/1.474784)
* Many-body dissipative particle dynamics (MDPD), source: [Warren, PRE, 2003](https://doi.org/10.1103/PhysRevE.68.066702)
* An experimental generalised many-body DPD with more freedom in defining local density

This implementation of DPD/MDPD for non-bonded particles only. 
For more complex simulations please refer to 


## Code versions
The default language is Python. Key bottlenecks involving most for loops
are written in Fortran and linked via f2py or enhanced with Numba (a JIT compiler).

There are ad hoc versions in pure Fortran and Julia, which are not developed anymore.

The recommended and most developed way to simulate is via the package `dpdsim`.

There are tools to compute density profiles and radial distribution functions
stored in `utils.py`.


## Example
Pure DPD with Fortran-optimised code:
```Python
from dpdsim import DPDSim
sim = DPDSim(implementation="fortran", steps=1000, thermo=100)
sim.create_particle_inputs(kind="pure")
sim.run()
```

Separate functions can be accessed as follows:
```Python
sim.compute_pe() # returns potential enerigy
sim.compute_ke() # returns kinetic energy
sim.compute_local_density() # computes MDPD local density
sim.compute_force() # computes force, virial and the stress tensor diagonal
```

## Requirements
* a Fortran compiler
* numpy, numba, docopt


## Performance
Fortran-optimised code is about an order of magnitude faster than Python with Numba.




