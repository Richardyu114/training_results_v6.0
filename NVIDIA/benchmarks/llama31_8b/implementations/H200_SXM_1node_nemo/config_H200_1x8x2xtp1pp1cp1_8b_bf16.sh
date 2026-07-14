#!/bin/bash
# MLPerf Training v6.0 - Llama 3.1 8B - single node 8x H200 (SXM/NVL, 141GB) - BF16
#
# Precision: pure BF16 mixed precision (no FP8). Comparison / fallback path to the FP8 config in
# this directory. Same parallelism and batch layout; only the numeric precision differs.
#
# NOTE: NOT yet validated on H200. Dell shipped only FP8 8xH200 results (XE7740), so the FP8 config
# has a reference run but this BF16 variant does not. It is the FP8 layout with the NeMo FP8 switch
# off; convergence and throughput on H200 must still be confirmed by our own run.
#
# Parallelism: pure data parallel. TP=PP=CP=1 -> DP=8.
#   GBS = MINIBS * (DGXNGPU * DGXNNODES) / (TP * PP * CP) = 2 * 8 / 1 = 16
#   grad_accum = MINIBS / MICRO_BATCH_SIZE = 2 / 2 = 1

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

# Parallelism
export TENSOR_MODEL_PARALLEL=1
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1
export SEQ_PARALLEL=False
export TP_COMM_OVERLAP=False

# Batch
export MINIBS=2
export MICRO_BATCH_SIZE=2

# Optimizer / schedule
export LR=0.0004
export WARMUP_STEPS=16
export VAL_CHECK_INTERVAL=768

# Disable FP8. config_common.sh sets FP8=True and FP8_PARAM_GATHER=True, and config_common_8b.sh
# sets FP8_HYBRID=True; all three must be turned off for a pure-BF16 run (fp8_param_gather with
# FP8 off is meaningless and would mis-shard the distributed optimizer). trainer.precision stays
# bf16 (the yaml default), so the optimizer/params run in BF16 mixed precision.
export FP8=False
export FP8_HYBRID=False
export FP8_PARAM_GATHER=False

# Verbose training log: emit Megatron-Bridge's native per-iteration line, including the built-in
# "throughput per GPU (TFLOP/s/GPU)" metric (same formula as the AMD Primus logs) + loss + step
# time, every LOG_EVERY_N_STEPS (=32, from config_common_8b.sh). Set to 0 for a clean MLPerf-style
# run (MLLOG only). Enabled by default since this is an internal enablement config.
export MLPERF_VERBOSE_LOGS=1

# Single node
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//')

export WALLTIME_RUNANDTIME=400
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
