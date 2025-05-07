#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --time=12-00:00:00
#SBATCH -p batch
#SBATCH --qos=long
#SBATCH --mem=0
#SBATCH --output=3_GNN.out
#SBATCH --error=3_GNN.err

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/GNN_method/

# Initialiser micromamba
eval "$(micromamba shell hook --shell bash)"

micromamba activate GP_for_cut_selection

echo "Début du job à $(date)"
echo "ID du job : ${SLURM_JOBID}"
echo "Répertoire de soumission : ${SLURM_SUBMIT_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

directory_path="ResultsGCNN/"
mkdir -p "$directory_path"
directory_path="Tensorboard/"
mkdir -p "$directory_path"

echo "training neural network"
python Slurm/train_neural_network.py TransformedInstances/Train/ TransformedInstances/Test/ TransformedSolutions/ Features RootResults/ ResultsGCNN/ Tensorboard/ TempFiles/GNN/ None Outfiles/ 250 0.1 20 667 False

echo "Evaluating trained network"
python Slurm/evaluate_trained_network.py TransformedInstances/Test TransformedSolutions Features/ RootResults/ ResultsGCNN/ Tensorboard/ TempFiles/GNN/ ResultsGCNN/actor.pt Outfiles True 

echo "ended of job at $(date)"

micromamba deactivate 
