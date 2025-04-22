#!/bin/bash -l
#SBATCH -c 28
#SBATCH -t 190
#SBATCH -p batch
#SBATCH --exclusive
cd
micromamba activate a_even_better_name_than_base
python -u node-selection-method-using-gp/subprocess_execution_MIPLIB.py