#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --time=1-00:00:00
#SBATCH -p batch
##SBATCH --qos=long
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

directory_path="Features/"
mkdir -p "$directory_path"
directory_path="ResultsGCNN/"
mkdir -p "$directory_path"
directory_path="Tensorboard/"
mkdir -p "$directory_path"
directory_path="TempFiles/GNN/"
mkdir -p "$directory_path"

echo "Generating features"
python Slurm/generate_feature_vectors.py TransformedInstances/All Features/ TempFiles/GNN/ Outfiles/ 1

echo "Generating training and test sets of the transformed data"
python generate_test_train_set.py data/ TransformedInstances/ 20 'train+test' True

echo "training neural network"
python Slurm/train_neural_network.py TransformedInstances/Train/ TransformedInstances/Test/ TransformedSolutions/ Features RootResults/ ResultsGCNN/ Tensorboard/ TempFiles/GNN/ None Outfiles/ 250 0.1 20 667 False

echo "Evaluating trained network"
python Slurm/evaluate_trained_network.py TransformedInstances/Test TransformedSolutions Features/ RootResults/ ResultsGCNN/ Tensorboard/ TempFiles/GNN/ ResultsGCNN/actor.pt Outfiles True 

echo "ended of job at $(date)"

micromamba deactivate 
