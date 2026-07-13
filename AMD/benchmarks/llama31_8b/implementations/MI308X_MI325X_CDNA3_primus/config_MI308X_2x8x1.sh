#!/bin/bash
# MLPerf LLama3.1 8B Configuration for MI308X (2x8x1) -- TWO-NODE / 16 GPU
# CDNA3 (gfx942) variant: FP8 hybrid instead of FP4/mxfp4 (no native FP4 on CDNA3).
# NOT a valid MLPerf closed submission config (precision changed) -- internal enablement only.
#
# This is the 2-node sibling of config_MI308X_1x8x1.sh. It is launched by
# run_with_docker_2node.sh, which sets NODE_RANK / MASTER_ADDR per node (do NOT hardcode them
# to a single node here). Everything except NNODES, the cross-node NCCL env, and the platform
# label matches the single-node config.
#
# torchrun world_size = GPUS_PER_NODE * NNODES = 8 * 2 = 16.

export DGXSYSTEM=MI308X_2x8x1
export GPUS_PER_NODE=8
export NNODES=2
# NODE_RANK and MASTER_ADDR are injected per-node by run_with_docker_2node.sh:
#   node0 -> NODE_RANK=0, node1 -> NODE_RANK=1, both -> MASTER_ADDR=<node0 IP>.
# The defaults below are only fallbacks so a bare `source` doesn't leave them unset; the launcher
# overrides them via env before calling run_with_docker.sh (both names are in the exported set that
# run_with_docker.sh forwards into the container).
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=29502

export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH}"
export EXP=/workspace/code/conf/llama3.1_8B-pretrain-fp8.yaml
export DATA_PATH=/data

export PRIMUS_MICRO_BATCH_SIZE=2
# GBS=64 at world_size=16 -> grad_acc = 64/(2*16) = 2 (same per-GPU grad-acc as the 8-GPU run).
# This is the point of going two-node: 16 GPUs process 2x the global batch per step, so reaching a
# given token budget takes ~half the steps (strong scaling / speedup). (GBS=32 would instead give
# grad_acc=1 — same global batch as single-node, faster per step but not fewer steps.)
# NOTE: doubling GBS usually warrants an LR re-tune; we keep LR conservative for now (CDNA3
# stability first) and treat LR-vs-batch scaling as a later optimization.
export PRIMUS_GLOBAL_BATCH_SIZE=64
export PRIMUS_LR=8e-4
export PRIMUS_MIN_LR=8e-5
# PRIMUS_TRAIN_ITERS: default 200 (short). For a two-node bring-up start with 50 (env override).
# Set 1200000 for full train-to-target. LR: use 3e-4 for full runs (8e-4 diverges on CDNA3).
export PRIMUS_TRAIN_ITERS=200
export PRIMUS_LR_WARMUP_ITERS=64
export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))  # Auto-computed

export HSA_ENABLE_INTERRUPT=0
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
export PRIMUS_APPLY_ROPE_FUSION=True

export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
# NVTE_CK_IS_V3_ATOMIC_FP32=1 (TE default) is REQUIRED on CDNA3 (gfx942). The MI350X submission
# set it to 0 (non-FP32 atomic accumulation in the CK v3 attention backward), which overflows to
# Inf on gfx942 with seq_length=8192 — training then hits "found Inf in local grad norm ... in
# backward pass" on the first step (both FP8 and BF16). Keep it at 1 here.
export NVTE_CK_IS_V3_ATOMIC_FP32=1
export NVTE_USE_AITER_ROPE=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_USE_RMSNORM_TRITON=0
export NVTE_LOG_CK_CONFIG=0
export NVTE_LOG_FUSED_ATTN_CONFIG=0
export USE_TE_SWIGLU=1

export ENABLE_TRANSPOSE_CACHE=1
export CK_FUSED_ATTN_LOG_CONFIG=0
export CHECK_FOR_NAN_IN_GRAD=0

export TOKENIZERS_PARALLELISM=false
export NCCL_CHECKS_DISABLE=1
export TORCH_NCCL_HIGH_PRIORITY=1

export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_TRAIN_LOSS_LOG_FREQ=0
export MLLOG_TARGET_EVAL_LOSS=3.3
export TARGET_EVAL_LOSS=3.3
export MLLOG_SUBMISSION_BENCHMARK=llama31_8b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI308X_2node
export MLLOG_TENSOR_PARALLELISM=1
export MLLOG_PIPELINE_PARALLELISM=1
export MLLOG_CONTEXT_PARALLELISM=1
export MLLOG_EXPERT_PARALLELISM=1
export MLLOG_MICRO_BATCH_SIZE=2
export MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR='fp8'

export FP8=true          # informational; real switch is `fp8: hybrid` in the yaml
export FP8_RECIPE=hybrid  # informational; 'hybrid' is the FP8 *format*, not the scaling recipe

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

export SYNTH_WARMUP_STEPS=20
export SYNTH_WARMUP_VALID_STEPS=10
export WARMUP_RECIPE=fp8_hybrid
export SYNTH_WARMUP_EMPTY_CACHE=1
export MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-0}

# --- NCCL tuning (kept aligned with config_MI308X_1x8x1.sh) ---
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=5
export NCCL_SINGLE_RING_THRESHOLD=0
export NCCL_BUFFSIZE=2097152

# --- CROSS-NODE NCCL/RoCE env (for 2-node). Auto-detected at source time so this config is
#     portable (no cluster-specific values hardcoded). Each can still be overridden by exporting
#     it before sourcing. To disable a knob, export it empty (e.g. NCCL_IB_HCA="").
#
#   NCCL_SOCKET_IFNAME : NIC carrying the default route (used for c10d rendezvous / bootstrap).
#   NCCL_IB_HCA        : RoCE HCAs for GPU-GPU RDMA. Auto-picks bnxt_re* if present, else mlx5*.
#   NCCL_IB_GID_INDEX  : RoCE v2 GID index (auto-picks the highest RoCE v2 index, usually the
#                        routable one; falls back to 3).
#   NCCL_NET_GDR_LEVEL : GPU Direct RDMA level (performance only; safe default 3).
#
# NOTE: auto-detection reflects the node where this config is *sourced*. In the 2-node launcher
# the config is sourced independently on each node, so each node detects its own NICs/HCAs.

# NIC with the default route (fallback: first non-lo UP interface)
_auto_ifname="$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' | head -1)"
[ -z "$_auto_ifname" ] && _auto_ifname="$(ip -o link show up 2>/dev/null | awk -F': ' '$2!="lo"{print $2; exit}')"
if [ -z "${NCCL_SOCKET_IFNAME+x}" ]; then
  NCCL_SOCKET_IFNAME="$_auto_ifname"
fi
export NCCL_SOCKET_IFNAME
# GLOO also needs the NIC pinned. Without this, gloo resolves the local address via `hostname -i`,
# which on Debian/Ubuntu returns 127.0.1.1 (a loopback alias in /etc/hosts) — cross-node ranks then
# try to reach each other at 127.0.1.1 and get "Connection refused". Pin gloo to the real NIC.
if [ -z "${GLOO_SOCKET_IFNAME+x}" ]; then
  GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$_auto_ifname}"
fi
export GLOO_SOCKET_IFNAME

# RoCE HCAs: prefer bnxt_re*, else mlx5*, else all IB devices
_auto_hca="$(ls /sys/class/infiniband/ 2>/dev/null | grep -E '^bnxt_re' | paste -sd, -)"
[ -z "$_auto_hca" ] && _auto_hca="$(ls /sys/class/infiniband/ 2>/dev/null | grep -E '^mlx5' | paste -sd, -)"
[ -z "$_auto_hca" ] && _auto_hca="$(ls /sys/class/infiniband/ 2>/dev/null | paste -sd, -)"
if [ -z "${NCCL_IB_HCA+x}" ]; then
  NCCL_IB_HCA="$_auto_hca"
fi
export NCCL_IB_HCA

# Highest RoCE v2 GID index on the first detected HCA (fallback 3)
_auto_gid=""
_first_hca="$(ls /sys/class/infiniband/ 2>/dev/null | grep -E '^bnxt_re|^mlx5' | head -1)"
if [ -n "$_first_hca" ]; then
  for _t in /sys/class/infiniband/"$_first_hca"/ports/1/gid_attrs/types/*; do
    [ -e "$_t" ] || continue
    if [ "$(cat "$_t" 2>/dev/null)" = "RoCE v2" ]; then _auto_gid="$(basename "$_t")"; fi
  done
fi
[ -z "$_auto_gid" ] && _auto_gid=3
if [ -z "${NCCL_IB_GID_INDEX+x}" ]; then
  NCCL_IB_GID_INDEX="$_auto_gid"
fi
export NCCL_IB_GID_INDEX

if [ -z "${NCCL_NET_GDR_LEVEL+x}" ]; then
  NCCL_NET_GDR_LEVEL=3
fi
export NCCL_NET_GDR_LEVEL

echo "[config] NCCL cross-node auto-detect: IFNAME=$NCCL_SOCKET_IFNAME GID=$NCCL_IB_GID_INDEX HCA=$NCCL_IB_HCA" >&2

export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA_OPTIMIZATIONS=1
export PYTORCH_ROC_ALLOC_CONF=expandable_segments:True
export HIP_API_BLOCKING=0
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
