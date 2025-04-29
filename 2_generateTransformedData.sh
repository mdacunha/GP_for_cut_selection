#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --time=0-12:00:00
#SBATCH -p batch
#SBATCH --output=2_Gen_transformed_data_output.out
#SBATCH --error=2_Gen_transformed_data_error.err

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/GNN_method/

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)"

micromamba activate GP_for_cut_selection

echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Generating transformed data"
python Slurm/generate_standard_data.py data/ TransformedSolutions/ TransformedInstances/All TransformedSolutions/ RootResults/ FullResults/ TempFiles/ Outfiles/ 1 False True True

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/
echo "removing non accepted instances from GP dataset for comparison on the same dataset"
python remove_non_accepted_instances.py GNN_method/TransformedInstances/All data/gisp/train data/gisp/test

echo "ended of job at $(date)"

micromamba deactivate 
