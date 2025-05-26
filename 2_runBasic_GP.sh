#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --time=9-00:00:00
#SBATCH -p batch
#SBATCH --qos=long
#SBATCH --output=3_Basic_GP.out
#SBATCH --error=3_Basic_GP.err

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/ || { echo "Failed to change directory"; exit 1; }

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)" || { echo "Failed to initialize Micromamba"; exit 1; }

micromamba activate GP_for_cut_selection || { echo "Failed to activate Micromamba environment"; exit 1; }

export SLURM_CPU_BIND=none

HOSTFILE=$(pwd)/hostfile
SCOOP_WRAPPER=$(pwd)/scoop-python.sh

cat << EOF > $SCOOP_WRAPPER
#!/bin/bash -l
eval "\$(micromamba shell hook --shell=bash)"
micromamba activate GP_for_cut_selection
EOF
echo 'python $@' >> $SCOOP_WRAPPER

chmod +x $SCOOP_WRAPPER

scontrol show hostnames > $HOSTFILE

echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

if [ -z "$1" ]; then
  echo "Arguments non fournis. Utilisation : sbatch script.sh <argument> <argument>"
  exit 1
fi

ARGUMENT1=$1

# Exécuter le script avec SCOOP
INPUTFILE=$(pwd)/basic_main.py
python -m scoop --hostfile $HOSTFILE -n ${SLURM_NTASKS} --python-interpreter=$SCOOP_WRAPPER $INPUTFILE "$ARGUMENT1" None

echo "Fin du job à $(date)"

micromamba deactivate
