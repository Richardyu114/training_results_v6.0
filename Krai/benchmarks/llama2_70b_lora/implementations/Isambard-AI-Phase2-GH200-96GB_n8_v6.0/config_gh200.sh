#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

: "${GPU_ARCH:=h}"
: "${CPUS_PER_TASK:=36}"

# hyperparameters
export MAX_STEPS=800
export LR=0.00055
export MINIBS=8
export TP=4
export PP=1
export CP=1

export FP8_ACT=1
export CG_WEIGHT_CACHING=1

# preventing OOM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8" 
export SP=1

export LAYER_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=1

export NCCL_NVLS_ENABLE=1

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=4
export BASE_TIME=96
#export WALLTIME_RUNANDTIME=$((96*$CP/$DGXNNODES)) #120
#export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
