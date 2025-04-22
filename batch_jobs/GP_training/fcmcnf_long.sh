#!/bin/bash -l
#SBATCH -c 16
#SBATCH -t20160
#SBATCH -p batch
#SBATCH --exclusive

cd
micromamba activate a_even_better_name_than_base
python -u node-selection-method-using-gp/genetic_programming_for_node_scoring.py -problem fcmcnf -nb_of_gen 50 -initial_pop 50 -parsimony_size 1.1 -fitness_size 3