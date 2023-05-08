#!/bin/bash

echo -e "\n===== Creating virtual environment..."
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo -e "\n===== Compiling Fortran modules..."
cd dpdsim/Fdpd
make
make f2py
cd ../..
echo -e "Compilation done."

fname=".bashrc.dpd"
cat > $HOME/$fname << EOF
# Updating pythonpath and path to include DPDsim

EOF

cdir=$(pwd)

echo 'export PYTHONPATH=$PYTHONPATH:'$cdir >>$HOME/$fname

echo "Add 'source $fname' to your .bashrc file to update PYTHONPATH"
