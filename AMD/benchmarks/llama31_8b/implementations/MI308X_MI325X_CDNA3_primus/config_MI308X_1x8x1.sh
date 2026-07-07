#!/bin/bash
# MLPerf LLama3.1 8B Configuration for MI308X (1x8x1)
# CDNA3 (gfx942) variant: FP8 hybrid instead of FP4/mxfp4 (no native FP4 on CDNA3).
# NOT a valid MLPerf closed submission config (precision changed) -- internal enablement only.

export DGXSYSTEM=MI308X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH}"
export EXP=/workspace/code/conf/llama3.1_8B-pretrain-fp8.yaml
export DATA_PATH=/data

export PRIMUS_MICRO_BATCH_SIZE=2
export PRIMUS_GLOBAL_BATCH_SIZE=32
export PRIMUS_LR=8e-4
export PRIMUS_MIN_LR=8e-5
# PRIMUS_TRAIN_ITERS: number of training steps. The MI350X submission uses 1200000 (full run to the
# target perplexity). We default to 200 for a short performance/enablement run (~a few minutes on
# 8 GPUs); set to 1200000 for full training, or e.g. 50 for a quick smoke test.
export PRIMUS_TRAIN_ITERS=200
export PRIMUS_LR_WARMUP_ITERS=64
export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))  # Auto-computed

export HSA_ENABLE_INTERRUPT=0
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
export PRIMUS_APPLY_ROPE_FUSION=True
# NOTE: Actual FP8 precision is controlled by `fp8: hybrid` in the yaml. The FP8_*
# env vars below are AMD-submission-script conveniences; Primus does not read them
# to set the recipe (scaling recipe defaults to `delayed`).

export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
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
export MLLOG_SUBMISSION_PLATFORM=MI308X
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
