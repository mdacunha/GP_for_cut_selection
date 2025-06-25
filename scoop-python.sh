#!/bin/bash -l
eval "$(micromamba shell hook --shell=bash)"
micromamba activate GP_for_cut_selection
python "$@"