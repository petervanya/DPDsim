#!/bin/bash

./dpd_sim.jl --config CONFIG --steps 50 --thermo 5 --dt 0.05 --gamma 4.5

rdf_general.py "Dump/dump_[4-5][0-9].xyz" --bt 1 --L 5
