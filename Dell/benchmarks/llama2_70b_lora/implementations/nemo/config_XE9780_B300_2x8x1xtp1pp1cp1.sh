#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=700
export LR=0.0004
export MINIBS=2
export CP=4
export MCORE_CUDA_GRAPH=1

#export SBATCH_NETWORK=${SBATCH_NETWORK:-sharp}
export USE_SHARP=$( [[ ${SBATCH_NETWORK:-} == "sharp" ]] && echo True || echo False )
export CUDNN_PATH=/usr/lib/x86_64-linux-gnu
# system parameters
export VBOOST_VALUE=0
export DGXNNODES=2
export DGXNGPU=8
export WALLTIME_RUNANDTIME=23
export NEXP=10
export WALLTIME=UNLIMITED
export NCCL_NVLS_ENABLE=1
export NCCL_IGNORE_COLLNET_MISMATCH=1
export NCCL_IB_HCA=^smi1
