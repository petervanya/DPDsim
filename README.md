# DPDsim
23/06/16

A simple implementation of dissipative particle dynamics method of non-bonded particles.
Original article: [Groot and Warren, JCP, 1997](http://dx.doi.org/10.1063/1.474784).

Written in Fortran (pure or via f2py), Python, and Julia.

Python package stored in `dpdsim`.

Python modules stored separately in `Pydpd`.

### Requirements
* Python3
* numpy, numba, docopt


### Performance
Fortran is about an order of magnitude faster than Python/Numba.


## TO DO
* [S] Generalise velocity-Verlet algo
* [M] Add pressure
* [M] Add stress tensor
* [L] Add Verlet/neighbour list (as option)
* [L] Add bonds
