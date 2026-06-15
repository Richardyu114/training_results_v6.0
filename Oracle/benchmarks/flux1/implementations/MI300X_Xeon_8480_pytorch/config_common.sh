export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export HYDRA_FULL_ERROR=1
export TQDM_DISABLE=True
export DISABLE_PRINT=${DISABLE_PRINT:-True} # filter out nemo prints. Set to False to debug.

export MODEL=${MODEL:-schnell}
export DATA=${DATA:-cc12m}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export EXP_NAME=${EXP_NAME:-flux-train-$(date +%y%m%d%H%M%S%N)}
# Validation interval in samples
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}

export GRAD_REDUCE_IN_FP32=${GRAD_REDUCE_IN_FP32:-False}
export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-False}
export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-False}
export USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-False}

export WARMUP_ENABLED=${WARMUP_ENABLED:-True}
export WARMUP_TRAIN_STEPS=${WARMUP_TRAIN_STEPS:-2}
export WARMUP_VALIDATION_STEPS=${WARMUP_VALIDATION_STEPS:-2}

export HF_HUB_OFFLINE=1 # disable network requests for HF transfers

# NCCL/RCCL configuration for AMD GPUs
export NCCL_NVLS_ENABLE=0
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0

# NCCL tuning for multinode performance
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32

# Enable InfiniBand/RDMA
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export UCX_NET_DEVICES=eth0
export NCCL_IB_SL=0
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_TIMEOUT=22
export RCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=10
export RCCL_IB_RETRY_CNT=10
export HCOLL_ENABLE_MCAST_ALL=0
export NCCL_IGNORE_CPU_AFFINITY=1
export RX_QUEUE_LEN=8192
export IB_RX_QUEUE_LEN=8192
export MODEL_TFLOP_PER_SAMPLE=20.3
