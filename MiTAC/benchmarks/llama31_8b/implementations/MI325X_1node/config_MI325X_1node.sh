#!/bin/bash
# MLPerf Training v6.0 - LLama 3.1 8B Pretraining
# Config: MI325X (CDNA3) 1-node x 8 GPUs
#
# Slimmed down to match AMD's official Primus MI300X (gfx942) FP8 config style:
#   examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml
#
# AMD's official config sets only `fp8: hybrid` in the YAML and lets Megatron
# defaults handle everything else. The previous v5.1-era env var pile
# (NVTE_FP8_DPA_BWD, NVTE_*, PRIMUS_FP8_RECIPE, WARMUP_RECIPE, NCCL tuning)
# was suspected to drive MI325X NaN at first eval. This file removes those
# overrides — anything still here is required by MLPerf compliance or by
# multi-node launch.

export DGXSYSTEM=MI325X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH:-}"
export EXP=/workspace/code/conf/llama3.1_8B-pretrain-fp8.yaml
export DATA_PATH=/data

# --- MLPerf v6.0 hyperparameters ---
# Fixed by §9 closed-division rule: train_iters=1.2M, betas/eps/weight_decay/etc (in YAML).
# Tunable but aligned with v6.0 RCP for GBS=32: LR=8e-4, warmup_samples=4096 (=128 iters at GBS=32).
# The 4 tunables use `:= default` so tuning/sweep can override via parent-shell env.
: "${PRIMUS_MICRO_BATCH_SIZE:=2}"
: "${PRIMUS_GLOBAL_BATCH_SIZE:=32}"
: "${PRIMUS_LR:=8e-4}"
: "${PRIMUS_LR_WARMUP_ITERS:=128}"                                               # GBS=32 RCP: 4096 warmup samples
export PRIMUS_MICRO_BATCH_SIZE PRIMUS_GLOBAL_BATCH_SIZE PRIMUS_LR PRIMUS_LR_WARMUP_ITERS
export PRIMUS_TRAIN_ITERS=1200000
export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))
# §9-coupled values — auto-derived to keep the closed-division invariants tight.
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS - PRIMUS_LR_WARMUP_ITERS))    # rule: train_iters - warmup_iters
export PRIMUS_MIN_LR=$(awk "BEGIN { printf \"%g\", ${PRIMUS_LR} * 0.1 }")        # rule: opt_base_learning_rate * 0.1

# --- ROCm runtime (kept; non-FP8) ---
export HSA_NO_SCRATCH_RECLAIM=1
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Precision flag (FP4 off; FP8 hybrid is set in the YAML, not env) ---
export FP4=false

# --- Logging ---
export AITER_LOG_LEVEL=ERROR
export AITER_LOG_MORE=0
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# --- MLPerf compliance logging (REQUIRED — these drive mllog output) ---
export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_TRAIN_LOSS_LOG_FREQ=0
export MLLOG_TARGET_EVAL_LOSS=3.3
export TARGET_EVAL_LOSS=3.3
export MLLOG_SUBMISSION_BENCHMARK=llama31_8b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=MiTAC
export MLLOG_SUBMISSION_PLATFORM=MI325X

# --- Synthetic warmup (allowed init phase, before timed training) ---
export SYNTH_WARMUP_STEPS=20
export SYNTH_WARMUP_VALID_STEPS=10
export SYNTH_WARMUP_EMPTY_CACHE=1

# -----------------------------------------------------------------------------
# REMOVED env vars (kept here as commented reference for archaeology — these
# came from the v5.1 reference run and are NOT in AMD's current official MI300X
# llama3.1_8B-FP8-pretrain config. Suspected as the trigger of MI325X FP8 NaN
# at first eval. Re-enable individually only if measured improvement.
# -----------------------------------------------------------------------------
# export PRIMUS_FP8_RECIPE=hybrid           # invalid Megatron value; use yaml fp8_recipe
# export FP8_PARAMS=1                       # silently triggered FP8 even when yaml said null
# export WARMUP_RECIPE=fp8_hybrid           # not in AMD official path
# export NVTE_FUSED_ATTN=1
# export NVTE_FUSED_ATTN_CK=1
# export NVTE_FUSED_ATTN_AOTRITON=1
# export NVTE_CK_USES_FWD_V3=1
# export NVTE_CK_USES_BWD_V3=1
# export NVTE_CK_IS_V3_ATOMIC_FP32=0
# export NVTE_USE_AITER_ROPE=1
# export NVTE_FP8_DPA_BWD=1
# export NVTE_USE_HIPBLASLT=1
# export NVTE_USE_CAST_TRANSPOSE_TRITON=1
# export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
# export NVTE_RS_STRIDED_ATOMIC=2
# export NVTE_ASYNC_AMAX_REDUCTION=1
# export NVTE_DP_AMAX_REDUCE_INTERVAL=0
# export NVTE_FLASH_ATTN=0
# export NVTE_USE_RMSNORM_TRITON=0
# export USE_TE_SWIGLU=1
# export USE_HIPBLASLT=1
# export TORCH_BLAS_PREFER_HIPBLASLT=1
# export PRIMUS_APPLY_ROPE_FUSION=True
# export ROCTRACER_LOG=1                    # not training-affecting; turn back on for profiling
# export ROCTRACER_LOG_LEVEL=5
# export HSA_ENABLE_INTERRUPT=0
# export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
# export HSA_ENABLE_SDMA=1
# NCCL/RCCL tuning — leave to defaults for now
# export NCCL_CHECKS_DISABLE=1
# export TORCH_NCCL_HIGH_PRIORITY=1
# export NCCL_NVLS_ENABLE=0
# export NCCL_MIN_P2P_NCHANNELS=32
# export NCCL_MIN_CTAS=32
# export NCCL_NCHANNELS_PER_NET_PEER=32
# TP overlap (TP=1 so these had no effect anyway)
# export TP_COMM_OVERLAP=False
# export MC_TP_OVERLAP_AG=False
# export MC_TP_OVERLAP_RS=False
# export MC_TP_OVERLAP_RS_DGRAD=False
# Profiler (off for submission)
# export TORCHPROF_OUTPUT_DIR=/results/artifacts
# export TORCHPROF_VERBOSE=0
# export TORCHPROF_MAXROWS=100
# export TORCHPROF_PROFILE_MEMORY=0
# export TORCHPROF_WITH_STACK=0
# export TORCHPROF_RECORD_SHAPES=0
# export TORCHPROF_WITH_FLOPS=0
# export PROF_WARMUP_STEPS=10
# export PROF_ACTIVE_STEPS=6372
