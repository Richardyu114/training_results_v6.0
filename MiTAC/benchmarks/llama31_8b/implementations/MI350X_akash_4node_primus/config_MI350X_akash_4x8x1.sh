#!/bin/bash
# MLPerf Training v6.0 - LLama 3.1 8B Pretraining
# Config: MI350X Akash (CDNA4) 4-node x 8 GPUs (32 GPUs total)
#
# Differs from MI350X 1-node:
#   - GBS=128, LR=2e-3, warmup=256 (v6.0 RCP for GBS=128)
#   - 4-node multi-host distributed launch (NODE_RANK / MASTER_ADDR set externally)
#   - 400G RoCE network (Pensando ionic on Akash variant)
#
# FP4 mxfp4 path on CDNA4. AMD-tuned a4w4 GEMM table required.

export DGXSYSTEM=MI350X_4x8x1
export GPUS_PER_NODE=8
export NNODES=4
export NODE_RANK=${NODE_RANK:-0}              # set per-node by orchestrator
export MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must be set for multi-node}
export MASTER_PORT=${MASTER_PORT:-29502}

export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH:-}"
export EXP=/workspace/code/conf/llama3.1_8B-pretrain-fp4_4node.yaml
export DATA_PATH=/data

# --- MLPerf v6.0 hyperparameters (aligned with v6.0 RCP for GBS=128) ---
# The 4 tunables use `:= default` so tuning/sweep can override via parent-shell env.
: "${PRIMUS_MICRO_BATCH_SIZE:=1}"                                                # RCP@GBS=128: grad_acc=4 → MBS=128/(4*32)=1
: "${PRIMUS_GLOBAL_BATCH_SIZE:=128}"                                             # 32 * 4 nodes
: "${PRIMUS_LR:=2e-3}"                                                           # GBS=128 RCP
: "${PRIMUS_LR_WARMUP_ITERS:=256}"                                               # GBS=128 RCP: 32,768 warmup samples
export PRIMUS_MICRO_BATCH_SIZE PRIMUS_GLOBAL_BATCH_SIZE PRIMUS_LR PRIMUS_LR_WARMUP_ITERS
export PRIMUS_TRAIN_ITERS=1200000
export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))
export EVAL_SAMPLES=1024   # MLPerf v6.0 closed_llama31_8b: must equal 1024
export PRIMUS_EVAL_ITERS=$((EVAL_SAMPLES / PRIMUS_GLOBAL_BATCH_SIZE))
# §9-coupled values — auto-derived to keep the closed-division invariants tight.
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS - PRIMUS_LR_WARMUP_ITERS))
export PRIMUS_MIN_LR=$(awk "BEGIN { printf \"%g\", ${PRIMUS_LR} * 0.1 }")

# --- ROCm runtime ---
export ROCTRACER_LOG=1
export ROCTRACER_LOG_LEVEL=5
export HSA_ENABLE_INTERRUPT=0
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Primus / Megatron precision: FP4 mxfp4 (CDNA4) ---
export PRIMUS_APPLY_ROPE_FUSION=True
export FP4=true
export FP4_RECIPE=mxfp4
export NVTE_MXFP4_USE_HADAMARD=1

# --- TransformerEngine FP4 path ---
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_USE_AITER_ROPE=1
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export USE_TE_SWIGLU=1
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export TOKENIZERS_PARALLELISM=false

# --- AITER a4w4 GEMM tuner (FP4-specific) ---
export AITER_CONFIG_GEMM_A4W4=/workspace/code/a4w4_tuned_gemms.csv

# --- NCCL/RCCL tunables (MI350X-akash multi-node, 8x Pensando Pollara 400 RoCE) ---
#
# Akash-rack NIC layout:
#   a1/a2/a4: Broadcom mgmt = ens512np0 (rocep51s0); 8 ionic = enp{6,22,102,118,134,150,230,246}s0
#   a3:       Broadcom mgmt = ens513np0 (rocep52s0); ionic NICs same as above
# NCCL_SOCKET_IFNAME is the bootstrap-only NIC; comma-list handles the a3 name.
# NCCL_IB_HCA='^rocep5' excludes both Broadcom variants from RDMA (prefix match).
# NCCL_IB_GID_INDEX=0 is required for Pollara 400 (link-local IPv6); IPv4-mapped
# (index 1) causes CQE error 12 with concurrent multi-NIC operations.
# NCCL_ALGO=Ring covers all (op, dtype) combos this RCCL build supports and is
# the fastest algo for the >=256MB grad all-reduce buckets training emits.
export NCCL_CHECKS_DISABLE=1
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_SOCKET_IFNAME=ens512np0,ens513np0
export NCCL_IB_HCA='^rocep5'
export NCCL_IB_GID_INDEX=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_MIN_NCHANNELS=112
export NCCL_ALGO=Ring
export NCCL_IGNORE_CPU_AFFINITY=1
export HIP_FORCE_DEV_KERNARG=1

# --- TP overlap (off; TP=1) ---
export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

# --- Logging ---
export AITER_LOG_LEVEL=ERROR
export AITER_LOG_MORE=0
export PYTHONWARNINGS=ignore

# --- MLPerf compliance logging ---
export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_TRAIN_LOSS_LOG_FREQ=0
export MLLOG_TARGET_EVAL_LOSS=3.3
export TARGET_EVAL_LOSS=3.3
export MLLOG_SUBMISSION_BENCHMARK=llama31_8b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=MiTAC
export MLLOG_SUBMISSION_PLATFORM=MI350X

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
