#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7
#SBATCH -G 1
#SBATCH -t2880
#SBATCH -p gpu

cd
module load lang/Python
micromamba activate node_select_gpu
cd node-selection-method-using-gp
python -u evaluation_gnn_gp_SCIPbaseline.py -problem gisp -partition transfer
python -u evaluation_gnn_gp_SCIPbaseline.py -problem gisp -partition test