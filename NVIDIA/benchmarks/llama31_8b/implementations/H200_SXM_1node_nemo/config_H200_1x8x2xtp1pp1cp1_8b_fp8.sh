#!/bin/bash
# MLPerf Training v6.0 - Llama 3.1 8B - single node 8x H200 (SXM/NVL, 141GB) - FP8
#
# Precision: FP8 hybrid (e4m3 fwd / e5m2 bwd), tensorwise scaling recipe. Attention runs in BF16
# (FP8_DPA stays False); only the linear GEMMs are FP8. This mirrors the Hopper-class reference
# (H200 has no native FP4, so the FP4 path used by Blackwell configs does not apply here).
#
# Parallelism: pure data parallel. TP=PP=CP=1 -> DP=8 (Llama-3.1-8B fits on one 141GB GPU).
#   GBS = MINIBS * (DGXNGPU * DGXNNODES) / (TP * PP * CP) = 2 * 8 / 1 = 16
#   grad_accum = MINIBS / MICRO_BATCH_SIZE = 2 / 2 = 1
#
# Values reproduce the verified single-node 8xH200 run (Dell XE7740 H200-NVL, MLPerf v6.0 results):
# GBS=16, MBS=2, LR=4e-4, warmup=16 steps (=256 samples at GBS 16), eval every 768 steps,
# converges to eval log-ppl <= 3.3 in ~9984 steps / ~4-4.4h wall clock.

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
export MINIBS=2              # per-DP-rank minibatch -> GBS = 2 * 8 = 16
export MICRO_BATCH_SIZE=2    # grad_accum = MINIBS / MBS = 1

# Optimizer / schedule
export LR=0.0004             # peak LR (MIN_LR defaults to LR/10 = 4e-5)
export WARMUP_STEPS=16       # 16 steps * GBS 16 = 256 warmup samples
export VAL_CHECK_INTERVAL=768

# FP8 hybrid is enabled via config_common.sh (FP8=True) + config_common_8b.sh (FP8_HYBRID=True);
# FP8_RECIPE=tensorwise and FP8_PARAM_GATHER=True also come from config_common.sh. Attention stays
# BF16 (FP8_DPA unset -> False). No override needed here; kept explicit for clarity:
export FP8=True
export FP8_HYBRID=True

# Verbose training log: emit Megatron-Bridge's native per-iteration line, including the built-in
# "throughput per GPU (TFLOP/s/GPU)" metric (same formula as the AMD Primus logs) + loss + step
# time, every LOG_EVERY_N_STEPS (=32, from config_common_8b.sh). Set to 0 for a clean MLPerf-style
# run (MLLOG only). Enabled by default since this is an internal enablement config.
export MLPERF_VERBOSE_LOGS=1

# Single node
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//')

# MAX_STEPS defaults to 1200000 (full run to target) from config_common_8b.sh.
# For a quick smoke test, override at launch, e.g.: MAX_STEPS=50 bash run_with_docker.sh
export WALLTIME_RUNANDTIME=400
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
