#!/bin/bash
# Llama 3.1 70B FP16 configuration for MI325X, two nodes (2x8 = 16 GPUs).
# TP=2 x PP=8 -> DP=1. GBS 64 = DP1 * MBS2 * 32 gradient-accumulation steps.
#
# Performance/enablement recipe. There is no llama31_70b MLPerf benchmark, so this is
# not a submission config: it runs a fixed number of steps and reports throughput.

export DGXSYSTEM=MI325X_2x8x1
export GPUS_PER_NODE=8
export NNODES=2
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29502}"

export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH}"
export EXP="${EXP:-/workspace/code/conf/llama3.1_70B-pretrain-fp16.yaml}"
export DATA_PATH=/data

# --- Parallel layout ---
export PRIMUS_TENSOR_PARALLEL_SIZE="${PRIMUS_TENSOR_PARALLEL_SIZE:-2}"
export PRIMUS_PIPELINE_PARALLEL_SIZE="${PRIMUS_PIPELINE_PARALLEL_SIZE:-8}"   # 80 layers / 8 = 10 layers per stage

# --- Batch ---
# world_size 16 / (TP2 * PP8) = DP 1. GBS 64 = DP1 * MBS2 * 32 accumulation steps.
export PRIMUS_MICRO_BATCH_SIZE=2
export PRIMUS_GLOBAL_BATCH_SIZE="${PRIMUS_GLOBAL_BATCH_SIZE:-64}"

# --- Schedule ---
export PRIMUS_LR="${PRIMUS_LR:-1e-5}"
export PRIMUS_MIN_LR="${PRIMUS_MIN_LR:-1e-6}"
# Short performance run. Measured: sec/iter differs 0.06% between 30 and 200 iters, so 100
# is already deep into steady state. Raise for a longer soak, lower to ~30 for a fast
# functional check.
export PRIMUS_TRAIN_ITERS="${PRIMUS_TRAIN_ITERS:-100}"
# Learning-rate warmup is expressed as a fraction of the schedule, not a step count, so it
# stays proportional when PRIMUS_TRAIN_ITERS changes. Megatron derives the step count itself
# (lr_warmup_steps = lr_warmup_fraction * lr_decay_steps, and lr_decay_iters == train_iters),
# so nothing needs to be computed here.
export PRIMUS_LR_WARMUP_FRACTION="${PRIMUS_LR_WARMUP_FRACTION:-0.1}"
if ! awk -v f="${PRIMUS_LR_WARMUP_FRACTION}" \
     'BEGIN { exit !(f ~ /^[0-9]*\.?[0-9]+$/ && f+0 >= 0 && f+0 <= 1) }'; then
    echo "ERROR: PRIMUS_LR_WARMUP_FRACTION must be between 0 and 1:" \
         "${PRIMUS_LR_WARMUP_FRACTION}" >&2
    return 2
fi
# Megatron asserts that only one of lr_warmup_fraction / lr_warmup_iters is set.
if [[ -n "${PRIMUS_LR_WARMUP_ITERS:-}" ]]; then
    echo "ERROR: PRIMUS_LR_WARMUP_ITERS is no longer used; set PRIMUS_LR_WARMUP_FRACTION" \
         "(fraction of train_iters, e.g. 0.1) instead" >&2
    return 2
fi
_lr_warmup_iters=$(awk -v n="${PRIMUS_TRAIN_ITERS}" -v f="${PRIMUS_LR_WARMUP_FRACTION}" \
                       'BEGIN { printf "%d", n * f }')
# Validation is disabled during the performance run (see the yaml).
export PRIMUS_EVAL_ITERS=0
export PRIMUS_EVAL_INTERVAL=1000000

export HSA_ENABLE_INTERRUPT=0
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
export PRIMUS_APPLY_ROPE_FUSION=True

export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# TransformerEngine fused attention / CK FA v3
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
# bf16 uses  fmha_fwd_v3, fp16 uses fmha_fwd_ck
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_FLASH_ATTN=0
# export AITER_LOG_LEVEL=ERROR

# acc
export NVTE_CK_IS_V3_ATOMIC_FP32=1
export NVTE_CK_HOW_V3_BF16_CVT=0

export NVTE_USE_AITER_ROPE=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_RMSNORM_TRITON=0

export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
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
# No convergence target for this run; 0.0 disables the target-reached logic.
export MLLOG_TARGET_EVAL_LOSS=0.0
export TARGET_EVAL_LOSS=0.0
export MLLOG_SUBMISSION_BENCHMARK=llama31_70b
export MLLOG_SUBMISSION_DIVISION=open
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI325X_2node
export MLLOG_TENSOR_PARALLELISM="${PRIMUS_TENSOR_PARALLEL_SIZE}"
export MLLOG_PIPELINE_PARALLELISM="${PRIMUS_PIPELINE_PARALLEL_SIZE}"
export MLLOG_CONTEXT_PARALLELISM=1
export MLLOG_EXPERT_PARALLELISM=1
export MLLOG_MICRO_BATCH_SIZE="${PRIMUS_MICRO_BATCH_SIZE}"
export MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR='fp16'

# Informational only; the yaml (`fp16: true`, `bf16: false`) is the runtime source of truth.
export FP8=false
export FP8_RECIPE=none

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# Synthetic-data warmup built into this harness: it compiles kernels and fills the memory
# pool before the timed region, and is unrelated to the learning-rate warmup above.
# WARMUP_RECIPE has no fp16 value; empty means "no autocast", which is what fp16 wants.
export SYNTH_WARMUP_STEPS=5
export SYNTH_WARMUP_VALID_STEPS=0
export WARMUP_RECIPE=""
export SYNTH_WARMUP_EMPTY_CACHE=1
export MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-1}

export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=5
export NCCL_SINGLE_RING_THRESHOLD=0
export NCCL_BUFFSIZE=2097152

export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA_OPTIMIZATIONS=1
export PYTORCH_ROC_ALLOC_CONF=expandable_segments:True
export HIP_API_BLOCKING=0
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# ---------------------------------------------------------------------------
# Batch / parallelism consistency checks.
# Data-parallel size is world_size/(TP*PP), NOT world_size: with TP2*PP8 on 16 GPUs DP=1.
# GBS must be a multiple of DP*MBS; the quotient is the gradient-accumulation count.
# ---------------------------------------------------------------------------
if [[ ! "${PRIMUS_GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PRIMUS_GLOBAL_BATCH_SIZE must be a positive integer:" \
         "${PRIMUS_GLOBAL_BATCH_SIZE}" >&2
    return 2
fi
_world_size=$((NNODES * GPUS_PER_NODE))
_model_parallel=$((PRIMUS_TENSOR_PARALLEL_SIZE * PRIMUS_PIPELINE_PARALLEL_SIZE))
if (( _world_size % _model_parallel != 0 )); then
    echo "ERROR: world_size=${_world_size} is not divisible by TP*PP=${_model_parallel}" >&2
    return 2
fi
_data_parallel=$((_world_size / _model_parallel))
if (( PRIMUS_GLOBAL_BATCH_SIZE % (_data_parallel * PRIMUS_MICRO_BATCH_SIZE) != 0 )); then
    echo "ERROR: PRIMUS_GLOBAL_BATCH_SIZE=${PRIMUS_GLOBAL_BATCH_SIZE} is not divisible by" \
         "DP*MBS=$((_data_parallel * PRIMUS_MICRO_BATCH_SIZE))" >&2
    return 2
fi
_grad_accum=$((PRIMUS_GLOBAL_BATCH_SIZE / (_data_parallel * PRIMUS_MICRO_BATCH_SIZE)))
# With PP=8 the pipeline runs (grad_accum + PP - 1) slots for grad_accum useful microbatches,
# so a small accumulation count wastes a large share of the pipeline.
if (( _grad_accum < PRIMUS_PIPELINE_PARALLEL_SIZE )); then
    echo "[config] WARNING: gradient accumulation ${_grad_accum} < PP ${PRIMUS_PIPELINE_PARALLEL_SIZE};" \
         "pipeline bubble will dominate" >&2
fi
echo "[config] ${DGXSYSTEM}: world=${_world_size} TP=${PRIMUS_TENSOR_PARALLEL_SIZE}" \
     "PP=${PRIMUS_PIPELINE_PARALLEL_SIZE} DP=${_data_parallel} MBS=${PRIMUS_MICRO_BATCH_SIZE}" \
     "GBS=${PRIMUS_GLOBAL_BATCH_SIZE} grad_accum=${_grad_accum}" >&2
echo "[config] ${DGXSYSTEM}: train_iters=${PRIMUS_TRAIN_ITERS}" \
     "lr_warmup_fraction=${PRIMUS_LR_WARMUP_FRACTION} (-> ~${_lr_warmup_iters} iters)" >&2

# ---------------------------------------------------------------------------
# Cross-node bootstrap and RoCE selection. Each node sources this config
# independently, so node-local interface/HCA names are discovered locally.
# ---------------------------------------------------------------------------

# NIC carrying the default route. Prefer iproute2 and fall back to the kernel
# route table; never guess from the first UP interface.
_auto_ifname=""
if command -v ip >/dev/null 2>&1; then
    _auto_ifname="$(
        ip route get 8.8.8.8 2>/dev/null \
            | awk '{for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}' \
            || true
    )"
fi
[[ -n "${_auto_ifname}" ]] || _auto_ifname="$(
    awk 'NR > 1 && $2 == "00000000" {print $1; exit}' /proc/net/route 2>/dev/null || true
)"
if [[ -z "${NCCL_SOCKET_IFNAME+x}" ]]; then
    if [[ -z "${_auto_ifname}" ]]; then
        echo "ERROR: cannot detect the default-route NIC; set NCCL_SOCKET_IFNAME explicitly" >&2
        return 2
    fi
    NCCL_SOCKET_IFNAME="${_auto_ifname}"
fi
export NCCL_SOCKET_IFNAME

if [[ -z "${GLOO_SOCKET_IFNAME+x}" ]]; then
    GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${_auto_ifname}}"
    if [[ -z "${GLOO_SOCKET_IFNAME}" ]]; then
        echo "ERROR: cannot detect the bootstrap NIC; set GLOO_SOCKET_IFNAME explicitly" >&2
        return 2
    fi
fi
export GLOO_SOCKET_IFNAME

# Prefer Broadcom HCAs by PCI vendor ID rather than device name: some MI325X
# systems expose Broadcom ports as rdma0..rdmaN instead of bnxt_re0..bnxt_reN.
_auto_hca="$(
    for _hca_path in /sys/class/infiniband/*; do
        [[ -e "${_hca_path}" ]] || continue
        _hca_vendor=$(cat "${_hca_path}/device/vendor" 2>/dev/null || true)
        if [[ "${_hca_vendor,,}" == 0x14e4 ]]; then
            basename "${_hca_path}"
        fi
    done | paste -sd, - || true
)"
[[ -n "${_auto_hca}" ]] \
    || _auto_hca="$(ls /sys/class/infiniband/ 2>/dev/null | grep -E '^mlx5' | paste -sd, - || true)"
[[ -n "${_auto_hca}" ]] \
    || _auto_hca="$(ls /sys/class/infiniband/ 2>/dev/null | paste -sd, - || true)"
if [[ -z "${NCCL_IB_HCA+x}" ]]; then
    NCCL_IB_HCA="${_auto_hca}"
fi
export NCCL_IB_HCA

# Highest RoCE v2 GID index on the first selected HCA (fallback: 3).
_auto_gid=""
_first_hca="${NCCL_IB_HCA%%,*}"
_first_hca="${_first_hca#=}"
_first_hca="${_first_hca%%:*}"
[[ -e "/sys/class/infiniband/${_first_hca}" ]] || _first_hca="${_auto_hca%%,*}"
if [[ -n "${_first_hca}" ]]; then
    for _gid_type in /sys/class/infiniband/"${_first_hca}"/ports/1/gid_attrs/types/*; do
        [[ -e "${_gid_type}" ]] || continue
        if [[ "$(cat "${_gid_type}" 2>/dev/null)" == "RoCE v2" ]]; then
            _auto_gid=$(basename "${_gid_type}")
        fi
    done
fi
[[ -n "${_auto_gid}" ]] || _auto_gid=3
if [[ -z "${NCCL_IB_GID_INDEX+x}" ]]; then
    NCCL_IB_GID_INDEX="${_auto_gid}"
fi
export NCCL_IB_GID_INDEX

if [[ -z "${NCCL_NET_GDR_LEVEL+x}" ]]; then
    NCCL_NET_GDR_LEVEL=3
fi
export NCCL_NET_GDR_LEVEL

echo "[config] ${DGXSYSTEM} network auto-detect: IFNAME=${NCCL_SOCKET_IFNAME}" \
     "GID=${NCCL_IB_GID_INDEX} HCA=${NCCL_IB_HCA}" >&2

unset _auto_gid _auto_hca _auto_ifname _data_parallel _first_hca _gid_type
unset _grad_accum _hca_path _lr_warmup_iters _hca_vendor _model_parallel _world_size
