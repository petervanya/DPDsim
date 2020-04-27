#!/bin/bash

echo "Compiling..."
cd dpdsim/Fdpd
make
make f2py
cd ../..

fname=".bashrc.dpd"
cat >$HOME/$fname<<EOF
# Updating pythonpath and path to include DPDsim

EOF

cdir=$(pwd)

echo 'export PYTHONPATH=$PYTHONPATH:'$cdir >>$HOME/$fname

echo "Add "source $fname" to your .bashrc file to update PYTHONPATH."
