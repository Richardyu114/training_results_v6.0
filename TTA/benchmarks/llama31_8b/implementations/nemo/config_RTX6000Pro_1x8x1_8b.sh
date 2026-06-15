# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# MLPerf Training v6.0 - LLaMA 3.1 8B (Closed Division)
# Hardware: 1x Node, 8x NVIDIA RTX PRO 6000 Blackwell (96GB, CC 12.0)

# Execution mode
export REMOTE=0
export USER="${USER:-training}"
export HOST="${HOSTNAME:-localhost}"
export ACCOUNT="mlperf"
export PARTITION="local"

# Hardware configuration
export NNODES=1
export GPUS_PER_NODE=8
export TIME="08:00:00"
export MAX_RETRIES=0

# Container and paths
export IMAGE="mlperf-nvidia:llama31_8b-pyt"
export JOB_DIR="/results"
export PREPROCESSED_PATH="/data/dataset"
export TOKENIZER_PATH="/data/tokenizer"
export TMP_NPY_INDEX="/data/npy_index"
export CONTINUAL_CKPT="/data/continual_ckpt"

# Model configuration
export SIZE="8b"
export USE_CKPT=0
export FROM_HF=0
export SAVE_CKPT=0

# Training hyperparameters (Closed Division compliant)
# HPs from HPE Cray XD685 (committee-recommended): BS=16, LR=4e-4, warmup_samples=256
# Precision: FP8 hybrid (default, matches original submission — no PRECISION env override)
# Note: MBS=1 fixed (MBS=2 OOMed on RTX PRO 6000 96GB). grad_accum = 16/(1*8) = 2.
export GBS=16                     # Global batch size (HPE ref_16)
export MBS=1                      # Micro batch size (fixed - MBS=2 fails)
export MAX_LR="4e-4"                # Peak learning rate (HPE ref_16)
export WARMUP_STEPS=16            # Warmup steps (16 * GBS=16 = 256 samples, HPE ref_16)
export MAX_STEPS=1200000          # Fixed max steps per MLPerf spec

# Evaluation
export EVAL_EVERY=12288           # Evaluate every 12288 sequences
export START_EVAL_AT=0            # Start evaluation from step 0

# Parallelism
export TENSOR_PARALLEL_SIZE=1
export NEXP=1
export NPAR=1
export START_STEPS="0"

# NCCL settings (TTA v5.1 RTX PRO 6000 reference)
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_MIN_NCHANNELS=4
export NCCL_MIN_CTAS=1
export NCCL_MAX_CTAS=8
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_WORK_FIFO_DEPTH=1048576
export NCCL_P2P_LEVEL=PIX
export NCCL_NET_GDR_LEVEL=OFF
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_HIGH_PRIORITY=1

# Compute / Memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_DEVICE_MAX_CONNECTIONS removed - allows multiple CUDA streams for better overlap

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
