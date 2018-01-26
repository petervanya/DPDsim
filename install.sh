#!/bin/bash

cd Fdpd
#make
#make f2py
cd ..

fname=".bashrc.dpd"
cat >$HOME/$fname<<EOF
# Updating pythonpath and path to include DPDsim

EOF

cdir=$(pwd)

echo 'export PYTHONPATH=$PYTHONPATH:'$cdir/Pydpd >>$HOME/$fname
echo 'export PYTHONPATH=$PYTHONPATH:'$cdir/Fname >>$HOME/$fname
echo "" >>$HOME/$fname
echo 'export PATH=$PATH':$cdir >>$HOME/$fname

echo "Add "source $fname" to your .bashrc file."
