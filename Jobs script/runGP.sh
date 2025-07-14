#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=127
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH --time=02-00:00:00
#SBATCH -p batch
##SBATCH --qos=aion-batch-long
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Vérification des arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
  echo "Arguments non fournis. Utilisation : sbatch run_gp_scoop.sh <arg1> <arg2> <arg3> <arg4> <arg5>"
  exit 1
fi

ARGUMENT1=$1
ARGUMENT2=$2
ARGUMENT3=$3
ARGUMENT4=$4
ARGUMENT5=$5

# Préparation des logs
LOGDIR=logs_ind_1_heuristic
mkdir -p "$LOGDIR"
if [ "$ARGUMENT2" == "RL" ]; then
  LOGFILE="${LOGDIR}/job_${ARGUMENT1}_${ARGUMENT2}_${ARGUMENT3}_${ARGUMENT4}_${ARGUMENT5}.out"
else
  LOGFILE="${LOGDIR}/job_${ARGUMENT1}_${ARGUMENT2}_${ARGUMENT3}.out"
fi
exec >"$LOGFILE" 2>&1

# Aller dans le répertoire de travail
cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/ || { echo "Failed to change directory"; exit 1; }

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)" || { echo "Failed to initialize Micromamba"; exit 1; }
micromamba activate GP_for_cut_selection || { echo "Failed to activate Micromamba environment"; exit 1; }

export SLURM_CPU_BIND=none
export SLURM_NTASKS=127

## Créer le wrapper Python
SCOOP_WRAPPER=$(pwd)/scoop-python.sh
cat << 'EOF' > "$SCOOP_WRAPPER"
#!/bin/bash -l
eval "$(micromamba shell hook --shell=bash)"
micromamba activate GP_for_cut_selection
python "$@"
EOF
chmod +x "$SCOOP_WRAPPER"

HOSTFILE=$(pwd)/hostfile
yes localhost | head -n "$SLURM_NTASKS" > "$HOSTFILE"

# Log infos job
echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

# Ajouter répertoire au PYTHONPATH si nécessaire
export PYTHONPATH=$(pwd):$PYTHONPATH

INPUTFILE=$(pwd)/main.py

# Lancer le script avec SCOOP
python -m scoop --hostfile "$HOSTFILE" -n "$SLURM_NTASKS" --python-interpreter="$SCOOP_WRAPPER" "$INPUTFILE" "$ARGUMENT1" "$ARGUMENT2" "$ARGUMENT3" "$ARGUMENT4" "$ARGUMENT5"

echo "Fin du job à $(date)"

micromamba deactivate
