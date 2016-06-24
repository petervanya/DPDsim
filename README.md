# DPDsim
23/06/16

Dissipative particle dynamics simulation of non-bonded particles.
Based on [http://dx.doi.org/10.1063/1.474784](Groot-Warren, JCP, 1997).

### Run
`./dpd_sim input.yaml`

### Requirements
* Python3
* `pip3 install numpy docopt yaml`


## TODO
* [L] Add Verlet/neighbour list (as option)
* [L] Add bonds
* [M] Add pressure
* [S] Generalise velocity-Verlet algo


## Testing
23/06/16
Strong divergence in temperature. Check force arrows, integration and dt.

24/06/16
Double-checked arrows of vectors, fixed random force. 
Temperature now stabilised, staying at about 3.0 kBT.
Next explore why not around 1.0 kBT.

