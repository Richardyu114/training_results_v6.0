#!/bin/bash
# Shared two-node overlay for the MI308X and MI325X single-node configs.
# This file is sourced by config_<platform>_2x8x1.sh; do not source it directly.

if [[ "${_TWO_NODE_PLATFORM:-}" != MI308X && "${_TWO_NODE_PLATFORM:-}" != MI325X ]]; then
    echo "ERROR: _TWO_NODE_PLATFORM must be MI308X or MI325X" >&2
    return 2
fi

_two_node_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
_base_config="${_two_node_dir}/../config_${_TWO_NODE_PLATFORM}_1x8x1.sh"
if [[ ! -r "${_base_config}" ]]; then
    echo "ERROR: single-node base config is not readable: ${_base_config}" >&2
    return 2
fi

# The single-node base owns all model and kernel tuning. Preserve the values
# injected by the two-node launcher while sourcing that base.
_two_node_rank="${NODE_RANK:-0}"
_two_node_master_addr="${MASTER_ADDR:-localhost}"
_two_node_master_port="${MASTER_PORT:-29502}"
_two_node_global_batch_size="${PRIMUS_GLOBAL_BATCH_SIZE:-32}"
_two_node_bf16_exp=/workspace/code/2nodes/conf/llama3.1_8B-pretrain-bf16.yaml
_two_node_fp8_exp=/workspace/code/2nodes/conf/llama3.1_8B-pretrain-fp8.yaml
_two_node_exp="${EXP:-${_two_node_fp8_exp}}"
_two_node_warmup_recipe="${WARMUP_RECIPE:-}"
_two_node_lr="${PRIMUS_LR:-3e-4}"
_two_node_min_lr="${PRIMUS_MIN_LR:-3e-5}"
_two_node_lr_warmup_iters="${PRIMUS_LR_WARMUP_ITERS:-64}"
_two_node_nvte_bf16_cvt_is_set=0
_two_node_nvte_bf16_cvt=""
if [[ -n "${NVTE_CK_HOW_V3_BF16_CVT+x}" ]]; then
    case "${NVTE_CK_HOW_V3_BF16_CVT}" in
        0|1|2)
            _two_node_nvte_bf16_cvt_is_set=1
            _two_node_nvte_bf16_cvt="${NVTE_CK_HOW_V3_BF16_CVT}"
            ;;
        *)
            echo "ERROR: NVTE_CK_HOW_V3_BF16_CVT must be 0, 1, or 2" >&2
            return 2
            ;;
    esac
fi
if [[ -z "${_two_node_warmup_recipe}" ]]; then
    case "${_two_node_exp##*/}" in
        llama3.1_8B-pretrain-bf16.yaml) _two_node_warmup_recipe=bf16 ;;
        llama3.1_8B-pretrain-fp8.yaml) _two_node_warmup_recipe=fp8_hybrid ;;
        *)
            echo "ERROR: set WARMUP_RECIPE when overriding EXP=${_two_node_exp}" >&2
            return 2
            ;;
    esac
fi
: "${PYTHONPATH:=}"
source "${_base_config}" || return $?

# Shared two-node training defaults. Snapshotting before the single-node base is sourced keeps
# direct wrapper use and the launcher consistent: FP8 is the default, while BF16 remains an
# explicit alternative.
export EXP="${_two_node_exp}"
export WARMUP_RECIPE="${_two_node_warmup_recipe}"
export PRIMUS_LR="${_two_node_lr}"
export PRIMUS_MIN_LR="${_two_node_min_lr}"
export PRIMUS_LR_WARMUP_ITERS="${_two_node_lr_warmup_iters}"

# These variables are informational in the CDNA3 base configs; the YAML remains the source of
# truth for training precision and scaling. Keep the metadata aligned with the standard 2-node
# experiment YAML selected above.
case "${EXP}" in
    "${_two_node_bf16_exp}")
        export FP8=false
        export FP8_RECIPE=none
        export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=bf16
        # Prefer conservative round-to-nearest-even for the standard BF16 recipe.
        if (( _two_node_nvte_bf16_cvt_is_set )); then
            export NVTE_CK_HOW_V3_BF16_CVT="${_two_node_nvte_bf16_cvt}"
        else
            export NVTE_CK_HOW_V3_BF16_CVT=0
        fi
        ;;
    "${_two_node_fp8_exp}")
        export FP8=true
        export FP8_RECIPE=e4m3
        export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=fp8
        # Use RTNE for the standard FP8 recipe while its run-to-run stability is evaluated.
        if (( _two_node_nvte_bf16_cvt_is_set )); then
            export NVTE_CK_HOW_V3_BF16_CVT="${_two_node_nvte_bf16_cvt}"
        else
            export NVTE_CK_HOW_V3_BF16_CVT=0
        fi
        ;;
    *)
        # A custom YAML owns its precise format/recipe. Keep only generic precision metadata here
        # instead of incorrectly labelling another same-named file as the standard E4M3 recipe.
        export FP8_RECIPE=custom
        case "${EXP##*/}" in
            *[Ff][Pp]8*)
                export FP8=true
                export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=fp8
                ;;
            *[Bb][Ff]16*)
                export FP8=false
                export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=bf16
                ;;
        esac
        ;;
esac

export DGXSYSTEM="${_TWO_NODE_PLATFORM}_2x8x1"
export NNODES=2
export NODE_RANK="${_two_node_rank}"
export MASTER_ADDR="${_two_node_master_addr}"
export MASTER_PORT="${_two_node_master_port}"

# world_size=16, MBS=2, GBS=32 -> gradient accumulation=1. The launcher forwards an explicit
# PRIMUS_GLOBAL_BATCH_SIZE override to both nodes.
if [[ ! "${_two_node_global_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PRIMUS_GLOBAL_BATCH_SIZE must be a positive integer: ${_two_node_global_batch_size}" >&2
    return 2
fi
export PRIMUS_GLOBAL_BATCH_SIZE="${_two_node_global_batch_size}"
if (( PRIMUS_GLOBAL_BATCH_SIZE % (NNODES * GPUS_PER_NODE * PRIMUS_MICRO_BATCH_SIZE) != 0 )); then
    echo "ERROR: PRIMUS_GLOBAL_BATCH_SIZE=${PRIMUS_GLOBAL_BATCH_SIZE} is not divisible by" \
         "NNODES*GPUS_PER_NODE*MBS=$((NNODES * GPUS_PER_NODE * PRIMUS_MICRO_BATCH_SIZE))" >&2
    return 2
fi
if (( EVAL_SAMPLES_INTERVAL % PRIMUS_GLOBAL_BATCH_SIZE != 0 )); then
    echo "ERROR: EVAL_SAMPLES_INTERVAL=${EVAL_SAMPLES_INTERVAL} is not divisible by" \
         "PRIMUS_GLOBAL_BATCH_SIZE=${PRIMUS_GLOBAL_BATCH_SIZE}" >&2
    return 2
fi
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))
export MLLOG_SUBMISSION_PLATFORM="${_TWO_NODE_PLATFORM}_2node"
export MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")

# Cross-node bootstrap and RoCE selection. Each node sources this config
# independently, so node-local interface/HCA names are discovered locally.

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

echo "[config] ${DGXSYSTEM} network auto-detect: IFNAME=${NCCL_SOCKET_IFNAME} GID=${NCCL_IB_GID_INDEX} HCA=${NCCL_IB_HCA}" >&2

unset _auto_gid _auto_hca _auto_ifname _base_config _first_hca _gid_type
unset _hca_path _hca_vendor _two_node_bf16_exp _two_node_dir _two_node_exp
unset _two_node_fp8_exp _two_node_global_batch_size _two_node_lr _two_node_min_lr
unset _two_node_lr_warmup_iters _two_node_nvte_bf16_cvt _two_node_nvte_bf16_cvt_is_set
unset _two_node_master_addr _two_node_master_port _two_node_rank _two_node_warmup_recipe
