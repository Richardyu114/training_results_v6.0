#!/bin/bash
# 4-node FP4 baseline probe: DGXNNODES=4, HEALING_ITER=170, STORE_GPU=True, FIRST_LAST_LAYERS_BF16=True
#
# 4-node: GBS = MINIBS × (DGXNGPU × DGXNNODES) / CP = 1 × 16 / 1 = 16
# VAL_CHECK_INTERVAL=384 samples → eval every 384/16 = 24 steps
# Expected first eval at step 96, convergence at step ~192 (3072 samples / 16 GBS)
# HEALING_ITER=170 → 22 FP8 steps before eval@192 (similar ratio as heal355 for 2-node)
# Probe to confirm convergence step and timing.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=550
export LR=0.0006
export MINIBS=1
export CP=1
export MCORE_CUDA_GRAPH=1
export NUM_WORKERS=4

export HEALING_ITER=170
export STORE_GPU=True
export FIRST_LAST_LAYERS_BF16=True

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=4
export DGXNGPU=4
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
