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
: "${PYTHONPATH:=}"
source "${_base_config}" || return $?

export DGXSYSTEM="${_TWO_NODE_PLATFORM}_2x8x1"
export NNODES=2
export NODE_RANK="${_two_node_rank}"
export MASTER_ADDR="${_two_node_master_addr}"
export MASTER_PORT="${_two_node_master_port}"

# world_size=16, MBS=2, GBS=64 -> gradient accumulation = 2.
export PRIMUS_GLOBAL_BATCH_SIZE=64
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
unset _hca_path _hca_vendor _two_node_dir _two_node_master_addr _two_node_master_port _two_node_rank
