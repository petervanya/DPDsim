# DPDsim
23/06/16

Dissipative particle dynamics simulation of non-bonded particles.
Based on [http://dx.doi.org/10.1063/1.474784](Groot, JCP, 1997).

Written in Fortran (pure or via f2py), Python, and Julia.
Fortran is about an order of magnitude faster than Python/Numba.

### Requirements
* Python3
* numpy, numba, docopt


## TO DO
* [L] Add Verlet/neighbour list (as option)
* [L] Add bonds
* [M] Add pressure
* [S] Generalise velocity-Verlet algo
