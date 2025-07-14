#!/bin/bash
#SBATCH --output=/dev/null

SCOOP_WRAPPER=$(pwd)/scoop-python.sh
cat << 'EOF' > "$SCOOP_WRAPPER"
#!/bin/bash -l
eval "$(micromamba shell hook --shell=bash)"
micromamba activate GP_for_cut_selection
python "$@"
EOF
chmod +x "$SCOOP_WRAPPER"

# Listes d'arguments
ARG1_LIST=("gisp" "wpsm" "fcmcnf") 
ARG2_LIST=("RL") #("2" "5" "10" "20" "30" "40" "50" "70" "heuristic" "RL")
ARG3_LIST=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14")
ARG4_LIST=("only_scores") #("only_features" "scores_and_features")
ARG5_LIST=("not_parallel") # "parallel")

# Soumettre un job par combinaison
for arg1 in "${ARG1_LIST[@]}"; do
  for arg2 in "${ARG2_LIST[@]}"; do
    if [ "$arg2" == "RL" ]; then
      for arg4 in "${ARG4_LIST[@]}"; do
        for arg3 in "${ARG3_LIST[@]}"; do
          for arg5 in "${ARG5_LIST[@]}"; do
            sbatch runGP.sh "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"
          done
        done
      done
    else
      for arg3 in "${ARG3_LIST[@]}"; do
        sbatch runGP.sh "$arg1" "$arg2" "$arg3" "None" "None"
      done
    fi
  done
done
