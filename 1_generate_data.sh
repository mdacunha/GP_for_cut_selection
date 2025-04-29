#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --time=0-00:05:00
#SBATCH -p batch
#SBATCH --output=1_Gen_data_output.out
#SBATCH --error=1_Gen_data_error.err

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)"

micromamba activate GP_for_cut_selection

echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Generating data"
python generate_instances.py

echo "Copying data to GNN folder"
python copy_instances_for_GNN.py data/gisp/train data/gisp/test GNN_method/data/

echo "ended of job at $(date)"

micromamba deactivate