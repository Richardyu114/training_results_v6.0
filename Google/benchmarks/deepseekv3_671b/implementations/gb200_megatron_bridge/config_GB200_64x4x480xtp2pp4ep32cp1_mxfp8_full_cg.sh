source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh

export MAX_STEPS=100
export LIMIT_VAL_BATCHES=1
export LR=0.000023238  # 0.000024 * sqrt(15360/16384)
export LR_WARMUP_ITERS=4
export VAL_CHECK_INTERVAL=1
export MOE_AUX_LOSS_COEFF=0.01
export TARGET_LOG_PPL=3.6

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# WALLTIME = 5 + 1*(1000+5) = 1010 > 16h
export WALLTIME_RUNANDTIME=1000
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# Run with real dataset.
export MOE_ROUTER_FORCE_LOAD_BALANCING=False

export MINIBS=480
export TENSOR_MODEL_PARALLEL=2
export PIPELINE_MODEL_PARALLEL=4
export INTERLEAVED_PIPELINE=4
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=32
export PIPELINE_LAYOUT="Et*4|(t*4|)*14tmL"

export DGXNNODES=64
export DGXNGPU=4
export SEGMENT=16

export RECOMPUTE_MODULES="mlp"
export CUDA_GRAPH_IMPLEMENTATION="local"
export CUDA_GRAPH_SCOPE="full_iteration"
export CUDA_GRAPH_WARMUP_STEPS=2
export OPTIMIZER_CG=True
export ON_DEVICE_CLIP_GRAD=True

# MoE fusion knobs
export MOE_PERMUTE_FUSION=True
export MOE_ROUTER_FUSION=True
export APPLY_ROPE_FUSION=True

export MOE_ROUTER_DTYPE="bf16"

# Overlap margins
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0

# MoE dispatcher knobs
export MOE_TOKEN_DISPATCHER_TYPE=flex
export MOE_FLEX_DISPATCHER_BACKEND=hybridep
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32
export USE_MNNVL=1

export BINDCMD="bindpcie --cpu=node"

export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1

# #export NCCL_P2P_NET_CHUNKSIZE=
export ENABLE_PERF_CONFIGS=True

# unset these two since offloading is not used.
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C

# fused grouped MLP
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
# disable single-param GEMM for now
# export MOE_SINGLE_GROUPED_WEIGHT=True
export CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL=True

# paged stashing
export MOE_PAGED_STASH=True
export MOE_EXPERT_RANK_CAPACITY_FACTOR=4
export MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CUDA=1.5
export MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CPU=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"
export NCCL_GRAPH_REGISTER=0

# perf knobs for TE
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVTE_EXT_MARGIN_SM=16

# 1F1B overlap
export OVERLAP_MOE_EXPERT_PARALLEL_COMM=True
export USE_DYNAMIC_COMP_STREAM=True

export WARMUP_TRAIN_STEPS=3
export WARMUP_VALIDATION_STEPS=3

export HIGH_PRIORITY_A2A_COMM_STREAM=True
export MOE_HYBRIDEP_NUM_SMS_PREPROCESSING=32
export DELAY_WGRAD_COMPUTE=True

# === NCCL / fabric / RoCE routing ===
export NCCL_MNNVL_ENABLE=1
export SHARP=False
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_HCA=rocep139s0,rocep140s0,rocep195s0,rocep196s0
# Parallelize bootstrap TCP
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
# More patience on connect retries (default 34 retries, 100ms base)
export NCCL_SOCKET_RETRY_CNT=50
export NCCL_SOCKET_RETRY_SLEEP_MSEC=200

export GLOO_SOCKET_IFNAME=enp0s3
export HYDRA_FULL_ERROR=1
export DISTRIBUTED_TIMEOUT_MINUTES=30
export NCCL_TEST=0

# Fix coalesced allgather
export TORCH_DISTRIBUTED_DEBUG=OFF

export SKIP_TRAIN_CG_PARAM_AG=True

# === Paths ===
export LOAD_CHECKPOINT=/checkpoints
# export HF_DIR=/home/xiaotongyang_google_com/hf_home
# export LOGDIR=/home/xiaotongyang_google_com/logs
# export DATADIR=/mnt/localssd/dataset
# export LOAD_CHECKPOINTS_DIR=/mnt/localssd/mbridge_ckpt

# Submission related
# export MLPERF_SUBMISSION_ORG="Google_1"
# export MLPERF_SUBMISSION_PLATFORM="GB200_NVL72"
# export SEED=3

