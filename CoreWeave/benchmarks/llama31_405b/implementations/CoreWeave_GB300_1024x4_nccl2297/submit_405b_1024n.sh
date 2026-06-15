#!/bin/bash
cd "$(dirname "$0")"
source config_GB300_1024x4x8xtp2pp8cp2_cg_fp4.sh

export CONT=/mnt/data/images/llama31_405b-arm-20260428.sqsh
export DATADIR=/mnt/data/llama31-405b/dataset
export LOAD_CHECKPOINTS_PATH=/mnt/data/llama31-405b/checkpoint
export LOAD_CHECKPOINT=/load_checkpoints/405b
export LOGDIR=/mnt/home/lyan/llama31-405b-1024n/logs
export NEXP=1
export NCCL_TEST=1

# RoCE overrides
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_NET_PLUGIN=none
export NCCL_IB_HCA=ibp
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_ADDR_FAMILY=AF_INET6
export NCCL_IB_ADDR_RANGE=fd02::/16
export NCCL_IB_TC=96
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_NET_LATENCY=50
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=12
export NCCL_IB_QPS_PER_CONNECTION=4
# Enable lightweight NCCL diagnostic so we can identify the actual stuck rank
# in the DATA_PARALLEL_GROUP_WITH_CP hangs that have plagued 1024N runs
# (5676, 6035, 6214). WARN level only logs problems, not noise.
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET
export NVIDIA_IMEX_CHANNELS=0
export PMIX_MCA_gds='^ds12'
export USE_LIBUV=1
export TORCH_USE_LIBUV=1
export NCCL_P2P_NET_CHUNKSIZE=1048576
export NCCL_NVLS_NCHANNELS=24
export NCCL_NVLSTREE_MAX_CHUNKSIZE=262144
export NCCL_NVLS_CHUNKSIZE=262144
export HYDRA_FULL_ERROR=1
export BINDCMD=""

mkdir -p $LOGDIR

sbatch --exclude=gb300-128-121,gb300-128-189,gb300-128-191,gb300-137-103,gb300-138-141,gb300-140-089,gb300-128-239,gb300-129-129,gb300-140-125,gb300-143-169,gb300-144-109,gb300-154-109,gb300-154-123,gb300-154-143 -N ${DGXNNODES} \
  --partition=hpc-high \
  --segment=16 \
  --gres=gpu:gb300:4 \
  --mem=0 \
  --time=${WALLTIME} \
  run.sub
