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

directory_path="TransformedSolutions/"
mkdir -p "$directory_path"
directory_path="TransformedInstances/All/"
mkdir -p "$directory_path"
directory_path="RootResults/"
mkdir -p "$directory_path"
directory_path="FullResults/"
mkdir -p "$directory_path"
directory_path="TempFiles/Generate/"
mkdir -p "$directory_path"
directory_path="Outfiles/"
mkdir -p "$directory_path"
directory_path="TempFiles/GNN/"
mkdir -p "$directory_path"
directory_path="Features/"
mkdir -p "$directory_path"

echo "Generating transformed data"
python Slurm/generate_standard_data.py data/ TransformedSolutions/ TransformedInstances/All TransformedSolutions/ RootResults/ FullResults/ TempFiles/Generate/ Outfiles/ 1 False True True

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/
echo "removing non accepted instances from GP dataset for comparison on the same dataset"
python remove_non_accepted_instances.py GNN_method/TransformedInstances/All data/""/train data/""/test

cd /mnt/aiongpfs/users/mdacunha/GP_for_cut_selection/GNN_method/
echo "Generating features"
python Slurm/generate_feature_vectors.py TransformedInstances/All Features/ TempFiles/GNN/ Outfiles/ 1

echo "Generating training and test sets of the transformed data"
python generate_test_train_set.py data/""/test  GNN_method/TransformedInstances/ 20 'train+test' True

echo "ended of job at $(date)"

micromamba deactivate 
