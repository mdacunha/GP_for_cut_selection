#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=50
#SBATCH -c 1
#SBATCH --time=0-10:00:00
#SBATCH -p batch
##SBATCH --qos=long

# Vérification des arguments
if [ -z "$1" "$2" "$3"]; then
  echo "Arguments non fournis. Utilisation : sbatch run_gp_scoop.sh <argument>"
  exit 1
fi

ARGUMENT1=$1
ARGUMENT2=$2
ARGUMENT3=$3

#SBATCH --output=logs/job_${ARGUMENT1}_${ARGUMENT2}_${ARGUMENT3}.out
#SBATCH --error=/dev/null

# Aller dans le répertoire de travail
cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/ || { echo "Failed to change directory"; exit 1; }

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)" || { echo "Failed to initialize Micromamba"; exit 1; }
micromamba activate GP_for_cut_selection || { echo "Failed to activate Micromamba environment"; exit 1; }

export SLURM_CPU_BIND=none
export SLURM_NTASKS=50

# Créer le wrapper Python avec environnement activé
SCOOP_WRAPPER=$(pwd)/scoop-python.sh
cat << 'EOF' > "$SCOOP_WRAPPER"
#!/bin/bash -l
eval "$(micromamba shell hook --shell=bash)"
micromamba activate GP_for_cut_selection
python "$@"
EOF
chmod +x "$SCOOP_WRAPPER"

# Créer un fichier hostfile à partir de SLURM
HOSTFILE=$(pwd)/hostfile
scontrol show hostnames > "$HOSTFILE"

# Log infos job
echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

# Ajouter répertoire au PYTHONPATH si nécessaire
export PYTHONPATH=$(pwd):$PYTHONPATH

INPUTFILE=$(pwd)/main.py

# Lancer le script avec SCOOP
python -m scoop --hostfile "$HOSTFILE" -n "$SLURM_NTASKS" --python-interpreter="$SCOOP_WRAPPER" "$INPUTFILE" "$ARGUMENT1" "$ARGUMENT2" "$ARGUMENT3" None

echo "Fin du job à $(date)"

micromamba deactivate
