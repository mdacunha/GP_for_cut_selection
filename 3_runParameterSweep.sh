#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --time=3-00:00:00
#SBATCH -p batch
#SBATCH --output=Param_sweep_output.out
#SBATCH --error=Param_sweep_error.err

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/GNN_method/

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)"

micromamba activate GP_for_cut_selection

echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

# Grid search on the transformed instances.
# The program tries all the possible combinations out of the 256 variable choices for the scoring rule, with granularity
# of 0.1 and sum 1.0. Takes a long time to run.
# Using just a few instances from the Test set, for testing purposes.

echo "parameter sweep"
python Slurm/parameter_sweep.py TransformedInstances/All TransformedSolutions/ Features/ RootResults/ FullResults/ FinalResults/ TempFiles/ Outfiles/ True 

echo "ended of job at $(date)"

micromamba deactivate 