#!/bin/bash

./dpd_sim.jl --config CONFIG --steps 100 --thermo 10 --dt 0.03

cd Dump
./add_hdr.sh
cd ..

rdf_general.py "Dump/dump_*" --bt 1 --L 5
