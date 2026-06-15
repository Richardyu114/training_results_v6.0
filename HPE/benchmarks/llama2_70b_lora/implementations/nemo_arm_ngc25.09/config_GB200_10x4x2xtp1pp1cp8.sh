#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.0005
export MINIBS=2
export CP=8
export MCORE_CUDA_GRAPH=1
export NUM_WORKERS=2

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=10
export DGXNGPU=4
export SEGMENT=$(( CP / DGXNGPU ))
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

