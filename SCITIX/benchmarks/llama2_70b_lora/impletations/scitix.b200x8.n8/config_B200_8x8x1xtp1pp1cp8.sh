#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=700
export LR=0.0006
#export MIN_LR=0.0001
#export WARMUP_STEPS=50
# 强制锁定全局批次大小
export MINIBS=1
export CP=4
export TP=2
export PP=1
export MCORE_CUDA_GRAPH=1
export NUM_WORKERS=8

#export MC_TP_OVERLAP_RS_DGRAD=True
#export TP_COMM_OVERLAP=True
export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1
# system parameters

export CUDA_DEVICE_MAX_CONNECTION=1
export VBOOST_VALUE=0
export SKIP_EVALS=3
export SHARP=True
export SBATCH_NETWORK=sharp
export DGXNNODES=8
export DGXNGPU=8
export SEGMENT=$(( CP / DGXNGPU ))
export WALLTIME_RUNANDTIME=23
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

