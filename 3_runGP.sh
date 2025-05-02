#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --time=2-00:00:00
#SBATCH -p batch
##SBATCH --qos=long
#SBATCH --output=3_GP.out
#SBATCH --error=3_GP.err

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)"

micromamba activate GP_for_cut_selection

echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

# Vérifier si un argument a été passé
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Arguments non fournis. Utilisation : sbatch script.sh <argument> <argument>"
  exit 1
fi

ARGUMENT1=$1
ARGUMENT2=$2

echo "run GP"
python main.py "$ARGUMENT1" "$ARGUMENT2" None

echo "Fin du job à $(date)"

micromamba deactivate