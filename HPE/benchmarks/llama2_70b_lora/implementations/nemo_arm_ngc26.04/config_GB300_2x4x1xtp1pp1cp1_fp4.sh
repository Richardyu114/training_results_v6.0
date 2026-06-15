#!/bin/bash
# heal360_bf16ends: STORE_GPU=True, HEALING_ITER=360, FIRST_LAST_LAYERS_BF16=True
#
# Motivation: job 2604 (HEAL=350) gave 8/10 @ step-384 = 5.98-6.00 min.
#   Step-384 runs are ~5s slower than NVIDIA 5.9 min target.
#   Healing 10 steps later saves: 10 × (1.02-0.72)s = ~3s → step-384 target ~356s = 5.93 min
#   But 10 fewer FP4 steps risks more step-432 outliers.
#   FIRST_LAST_LAYERS_BF16=True keeps first+last transformer layers in BF16 during FP4 phase
#   → stabilizes convergence to offset the later heal risk.
#
# Expected step-384 timing:
#   360×0.72 + 24×1.02 + 2 (heal) + ~30 (overhead/evals) = 259 + 24 + 32 = ~315s = 5.25 min

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=550
export LR=0.0006
export MINIBS=1
export CP=1
export MCORE_CUDA_GRAPH=1
export NUM_WORKERS=4

export HEALING_ITER=360
export STORE_GPU=True
export FIRST_LAST_LAYERS_BF16=True

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=2
export DGXNGPU=4
export WALLTIME_RUNANDTIME=35
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
