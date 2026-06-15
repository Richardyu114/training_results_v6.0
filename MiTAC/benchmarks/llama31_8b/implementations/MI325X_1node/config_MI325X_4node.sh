#!/bin/bash
# MLPerf Training v6.0 - LLama 3.1 8B Pretraining
# Config: MI325X (CDNA3) 4-node x 8 GPUs (32 GPUs total)
#
# MLPerf v6.0 closed division — multi-node hyperparameters:
#   global_batch_size, opt_base_learning_rate, opt_learning_rate_warmup_steps are tunable
#   but must align with the published v6.0 RCP for the chosen GBS to pass the t-test.
#   For GBS=128 the RCP is: LR=2e-3, warmup_samples=32,768 (=256 iters at GBS=128).
#   See: https://raw.githubusercontent.com/mlcommons/logging/master/mlperf_logging/rcp_checker/training_6.0.0/rcps_llama31_8b.json
#
# Same hardware-tuning notes as config_MI325X_1x8x1.sh (FP8 hybrid, CDNA3 path).

export DGXSYSTEM=MI325X_4x8x1
export GPUS_PER_NODE=8
export NNODES=4
export NODE_RANK=${NODE_RANK:-0}              # set per-node by orchestrator
export MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must be set for multi-node}
export MASTER_PORT=${MASTER_PORT:-29502}

export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH:-}"
export EXP=/workspace/code/conf/llama3.1_8B-pretrain-fp8.yaml
export DATA_PATH=/data

# --- MLPerf v6.0 hyperparameters (aligned with v6.0 RCP for GBS=128) ---
# The 4 tunables use `:= default` so tuning/sweep can override via parent-shell env.
: "${PRIMUS_MICRO_BATCH_SIZE:=2}"
: "${PRIMUS_GLOBAL_BATCH_SIZE:=128}"                                             # 32 * 4 nodes
: "${PRIMUS_LR:=2e-3}"                                                           # GBS=128 RCP
: "${PRIMUS_LR_WARMUP_ITERS:=256}"                                               # GBS=128 RCP: 32,768 warmup samples
export PRIMUS_MICRO_BATCH_SIZE PRIMUS_GLOBAL_BATCH_SIZE PRIMUS_LR PRIMUS_LR_WARMUP_ITERS
export PRIMUS_TRAIN_ITERS=1200000
export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))
# §9-coupled values — auto-derived to keep the closed-division invariants tight.
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS - PRIMUS_LR_WARMUP_ITERS))    # rule: train_iters - warmup_iters
export PRIMUS_MIN_LR=$(awk "BEGIN { printf \"%g\", ${PRIMUS_LR} * 0.1 }")        # rule: opt_base_learning_rate * 0.1

# --- ROCm runtime ---
export ROCTRACER_LOG=1
export ROCTRACER_LOG_LEVEL=5
export HSA_ENABLE_INTERRUPT=0
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Primus / Megatron precision: FP8 hybrid (CDNA3) ---
export PRIMUS_APPLY_ROPE_FUSION=True
export PRIMUS_FP8_RECIPE=hybrid
export FP4=false
export FP8_PARAMS=1

# --- TransformerEngine FP8 path ---
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_USE_AITER_ROPE=1
export NVTE_FP8_DPA_BWD=1
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_FLASH_ATTN=0
export NVTE_USE_RMSNORM_TRITON=0
export USE_TE_SWIGLU=1
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export TOKENIZERS_PARALLELISM=false

# --- NCCL/RCCL tunables (MI325X multi-node, from v5.1 RoCE-validated run) ---
export NCCL_CHECKS_DISABLE=1
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_NVLS_ENABLE=0
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
# NCCL_SOCKET_IFNAME / NCCL_IB_HCA are intentionally unset here so the run script
# can auto-discover RoCE on bnxt_re. Set in env if discovery fails.

# --- TP overlap (off; TP=1) ---
export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

# --- Logging ---
export AITER_LOG_LEVEL=ERROR
export AITER_LOG_MORE=0
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1

# --- MLPerf compliance logging ---
export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_TRAIN_LOSS_LOG_FREQ=0
export MLLOG_TARGET_EVAL_LOSS=3.3
export TARGET_EVAL_LOSS=3.3
export MLLOG_SUBMISSION_BENCHMARK=llama31_8b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=MiTAC
export MLLOG_SUBMISSION_PLATFORM=MI325X

# --- Profiler (off for submission runs) ---
export TORCHPROF_OUTPUT_DIR=/results/artifacts
export TORCHPROF_VERBOSE=0
export TORCHPROF_MAXROWS=100
export TORCHPROF_PROFILE_MEMORY=0
export TORCHPROF_WITH_STACK=0
export TORCHPROF_RECORD_SHAPES=0
export TORCHPROF_WITH_FLOPS=0
export PROF_WARMUP_STEPS=10
export PROF_ACTIVE_STEPS=6372

# --- Synthetic warmup ---
export SYNTH_WARMUP_STEPS=20
export SYNTH_WARMUP_VALID_STEPS=10
export WARMUP_RECIPE=fp8_hybrid
export SYNTH_WARMUP_EMPTY_CACHE=1
