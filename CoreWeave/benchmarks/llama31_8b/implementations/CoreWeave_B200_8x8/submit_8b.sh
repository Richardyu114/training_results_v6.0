#!/bin/bash
# Launch llama31_8b on B200 8x8. Run from the dir holding run.sub + config_*.sh:
#   cd "$(dirname "$0")" && bash submit_8b.sh
set -euo pipefail

export NEXP=10
export DATADIR=/mnt/data/llama31-8b/dataset
export LOGDIR=$PWD/logs
mkdir -p "$LOGDIR"

source config_DGXB200_8x8x1xtp1pp1cp1_8b_fp4.sh
echo "[llama31_8b] NEXP=$NEXP DGXNNODES=$DGXNNODES WALLTIME=$WALLTIME"

CONT=/mnt/data/images/llama31_8b-amd-20260507.sqsh \
  sbatch --parsable --export=ALL \
         -p hpc-high -N 8 --gres=gpu:b200:8 \
         -t $((WALLTIME+30)) \
         run.sub
