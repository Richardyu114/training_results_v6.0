#!/bin/bash
# ==============================================================================
# MRC Environment Setup for GB200 Cluster (v6.0)
# Source this file before running jobs: source v6.0/configs/config_mrc_env.sh
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MLPERF_TRAIN_DIR="${REPO_ROOT}/MLPERF_train"

if [ ! -d "${MLPERF_TRAIN_DIR}" ]; then
    echo "ERROR: mlperf_train directory not found at ${MLPERF_TRAIN_DIR}"
    echo "Please ensure mlperf_train is in your repo root."
    return 1
fi

echo "Using MLPERF_TRAIN_DIR: ${MLPERF_TRAIN_DIR}"

# ==============================================================================
# MRC Library Paths
# ==============================================================================
export VMRC_LIBMRC_SO="/opt/mellanox/doca/lib/aarch64-linux-gnu/libnv_mrc.so"
export VMRC_LIBIBVERBS_SO="/lib/aarch64-linux-gnu/libibverbs.so.1"
export NCCL_VMRC_SO="/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/libibverbs.so"

# ==============================================================================
# MRC-Specific NCCL/UCX Settings
# ==============================================================================
export UCX_TLS=tcp
export UCX_NET_DEVICES=enP22p1s0f1

export NCCL_SOCKET_IFNAME=enP22p1s0f1
export GLOO_SOCKET_IFNAME=enP22p1s0f1
export NCCL_NET_PLUGIN=none
export NCCL_IB_HCA=mlx5_1,mlx5_0,mlx5_3,mlx5_2
export NCCL_IB_ECE_ENABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_IB_TC=4
export NCCL_IB_FIFO_TC=12
export NV_MRC_POST_SEND_PREFER_BF=1
export NCCL_MAX_NCHANNELS=4

export NCCL_MNNVL_ENABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_SHARP_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export NCCL_DEBUG=INFO

# ==============================================================================
# Slurm Large-Scale Launch Settings
# ==============================================================================
export SLURM_TREE_WIDTH=64
export SLURM_TCP_TIMEOUT=10
export SLURM_EIO_TIMEOUT=120
export SRUN_WAIT=120
module load mpi/hpcx
export SLURM_MPI_TYPE=pmix

# ==============================================================================
# Disable NCCL_TEST in the container
# ==============================================================================
export NCCL_TEST=0

# ==============================================================================
# Container Mount Paths
# ==============================================================================
export MRC_NCCL_MOUNT="${MLPERF_TRAIN_DIR}/mrc_packages/nccl_mrc/build/lib:/opt/nccl_mrc/build/lib:ro"
export MRC_HPCX_MOUNT="${MLPERF_TRAIN_DIR}/mrc_packages/hpcx-v2.23.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64/:/opt/hpcx-v2.23.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64/:ro"
export MRC_VERBS_MOUNT="/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/:/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/:ro"
export MRC_IMEX_MOUNT="/dev/nvidia-caps-imex-channels/channel0:/dev/nvidia-caps-imex-channels/channel0"
export MRC_DEV_MOUNT="/dev:/dev"

export EXTRA_MOUNTS="${MRC_DEV_MOUNT},${MRC_NCCL_MOUNT},${MRC_HPCX_MOUNT},${MRC_VERBS_MOUNT},${MRC_IMEX_MOUNT}"

# ==============================================================================
# Additional mounts from mrc_mounts.sh (if exists)
# ==============================================================================

MRC_MOUNTS_FILE="${MRC_MOUNTS_FILE:-${INFRA_DIR}/mrc_mounts.sh}"
if [ -f "${MRC_MOUNTS_FILE}" ]; then
    echo "Loading additional mounts from ${MRC_MOUNTS_FILE}..."
    source "${MRC_MOUNTS_FILE}"
    if [ -n "${MRC_MOUNTS:-}" ]; then
        export EXTRA_MOUNTS="${EXTRA_MOUNTS},${MRC_MOUNTS}"
    fi
fi

# ==============================================================================
# Mount host NVIDIA/CUDA driver libraries into container
# ==============================================================================
NVIDIA_CUDA_MOUNTS=""
for pattern in /usr/lib/aarch64-linux-gnu/libnvidia* /usr/lib/aarch64-linux-gnu/libcuda*; do
    for f in $pattern; do
        [ -f "$f" ] || continue
        case "$f" in *libcudart.so*) continue ;; esac
        if [ -z "$NVIDIA_CUDA_MOUNTS" ]; then
            NVIDIA_CUDA_MOUNTS="${f}:${f}:ro"
        else
            NVIDIA_CUDA_MOUNTS="${NVIDIA_CUDA_MOUNTS},${f}:${f}:ro"
        fi
    done
done
if [ -n "$NVIDIA_CUDA_MOUNTS" ]; then
    export EXTRA_MOUNTS="${EXTRA_MOUNTS},${NVIDIA_CUDA_MOUNTS}"
    echo "Added $(echo ${NVIDIA_CUDA_MOUNTS} | tr ',' '\n' | wc -l) NVIDIA/CUDA driver library mounts"
fi

# ==============================================================================
# Container Environment - LD_LIBRARY_PATH and PATH updates
# ==============================================================================
export MRC_LD_LIBRARY_PATH="/opt/nccl_mrc/build/lib:/opt/mellanox/doca/lib/aarch64-linux-gnu:/opt/mellanox/dpdk/lib/aarch64-linux-gnu:/opt/mellanox/flexio/lib:/opt/hpcx-v2.23.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64/ompi/lib:/usr/local/cuda/lib64:/lib/aarch64-linux-gnu/:/usr/lib/aarch64-linux-gnu/"
export MRC_PATH="/opt/hpcx-v2.23.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64/ompi/bin"

echo ""
echo "MRC environment configured successfully!"