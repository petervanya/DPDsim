# DPDsim
23/06/16

Dissipative particle dynamics simulation of non-bonded particles.
Based on [http://dx.doi.org/10.1063/1.474784](Groot, JCP, 1997).

### Run
```
./gen_input.py
./dpd_sim input.yaml
```

### Requirements
* Python3
* `pip3 install numpy docopt yaml`


## TO DO
* [L] Add Verlet/neighbour list (as option)
* [L] Add bonds
* [M] Add pressure
* [S] Generalise velocity-Verlet algo


