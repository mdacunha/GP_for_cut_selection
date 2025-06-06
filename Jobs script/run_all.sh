#!/bin/bash
#SBATCH --output=/dev/null

# Listes d'arguments
ARG1_LIST=("gisp" "wpsm" "fcmcnf")
ARG2_LIST=("5" "10" "30" "50" "70")
ARG3_LIST=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

# Soumettre un job par combinaison
for arg1 in "${ARG1_LIST[@]}"; do
  for arg2 in "${ARG2_LIST[@]}"; do
    for arg3 in "${ARG3_LIST[@]}"; do
      sbatch runGP.sh "$arg1" "$arg2" "$arg3"
    done
  done
done
