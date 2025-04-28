#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --time=0-12:00:00
#SBATCH -p batch
#SBATCH --output=SMAC_output.out
#SBATCH --error=SMAC_error.err

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/GNN_method/

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)"

micromamba activate GP_for_cut_selection

echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "SMAC"
python Slurm/smac_runs.py TransformedInstances/Train/ TransformedInstances/Test/ TransformedSolutions/ RootResults/ SmacResults/ TempFiles Outfiles 250 667

echo "ended of job at $(date)"

micromamba deactivate 