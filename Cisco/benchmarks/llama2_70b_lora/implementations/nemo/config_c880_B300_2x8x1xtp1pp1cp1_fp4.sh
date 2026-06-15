#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=550
export LR=0.0006
export MINIBS=1
export CP=2
export MCORE_CUDA_GRAPH=1
#export FP8_ACT=1
export NCCL_NVLS_ENABLE=0
export NUM_WORKERS=4

export HEALING_ITER=350
#export SBATCH_NETWORK=ethernet
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=2
export DGXNGPU=8
export WALLTIME_RUNANDTIME=25
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

