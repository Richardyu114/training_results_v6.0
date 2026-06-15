: "${NEXP:=10}"
export TRANSFORMERS_CACHE="/tmp/"
export TRITON_HOME="/tmp/"
export HF_HOME="/tmp/"
export REMOUNT_WORKDIR="$(dirname ${BASH_SOURCE[0]})/"

: "${PREPROCESSED_PATH:?PREPROCESSED_PATH not set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH not set}"
: "${SPM:?SPM not set}"
: "${CONT:?CONT not set}"
export PREPROC_DATA=$PREPROCESSED_PATH
export TOKENIZER=$TOKENIZER_PATH
export LOGDIR="${LOGDIR:-./results}"

#  Model defaults (8b)                                                     
export MODEL_SIZE="8b"
export TARGET_LOG_PPL="3.3"
export LOAD_CHECKPOINT=""
export OVERWRITTEN_NUM_LAYERS=32
export VAL_SAMPLES=1024
export MAX_STEPS=1200000
export OPT_LR_DECAY_STEPS=${MAX_STEPS}
export MICRO_BATCH_SIZE=1
export LOG_EVERY_N_STEPS=8
export GC_TRAIN=10000
export GC_VALID=10000
export TRAIN_ONLY=0
export LEGACY_DATASET=True

: "${LOAD_MINIMAL_NUM_SAMPLES:=0}"
if [[ "${LOAD_MINIMAL_NUM_SAMPLES}" -eq 1 ]]; then
  export MAX_STEPS=500
  export OVERRIDE_ZERO_CONSUMED_SAMPLES=0
  export INIT_GLOBAL_STEP=0
fi

if [[ "${NO_CKPT:-0}" -eq 1 ]]; then
    export LOAD_CHECKPOINT=""
    export CHECK_COMPLIANCE="0"
fi

# Distributed optimizer / DDP overlap 
export USE_DIST_OPTIMIZER=True
export OVERLAP_GRAD_REDUCE=True
export OVERLAP_PARAM_GATHER=True
# Cuda graphs require the optim step / wgrad defer overlaps to be off.
export OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=False
export DEFER_EMBEDDING_WGRAD_COMPUTE=False
export ALIGN_PARAM_GATHER=False
export WGRAD_DEFERRAL_LIMIT=50

# Transformer Engine
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_NORM_FWD_USE_CUDNN=0
export NVTE_NORM_BWD_USE_CUDNN=0
export NVTE_TP_OVERLAP_TRAINING_ONLY=1
# The TE op-fuser path mis-registers the FP8 delayed-scaling amax buffer and
# is incompatible with the delayed recipe; use the module-based TE path.
export USE_TE_OPS=False
export FUSED_QKV_ROPE=True
export CE_FUSION_IMPL=te
export TE_UB_ATOMIC_GEMM_RS=0
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True

# FP8 (delayed scaling: per-tensor amax tracked across steps).
export FP8=True
export FP8_HYBRID=True
export FP8_RECIPE="delayed"
export FP8_AMAX_HISTORY=1024
export FP8_AMAX_ALGO="max"
export FP8_PARAM_GATHER=True
export FP8_PARAMS=True

# FP8 attention (DPA). FP8_DPA is incompatible with the delayed FP8 recipe
# and is left in fp16/bf16.
export FP8_DPA=False
export NVTE_DPA_FP8_FORMAT="HYBRID"
export NVTE_DPA_FP8DS_AMAX_ALGO="most_recent"
export NVTE_DPA_FP8DS_AMAX_HISTLEN=1
export NVTE_DPA_FP8DS_REDUCE_AMAX=1
export NVTE_DPA_FP8CS_O_in_F16=1
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1

# CUDA graphs 
export FULL_CUDA_GRAPH=1
export MCORE_CUDA_GRAPH=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NCCL 
export NCCL_NVLS_ENABLE=0
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0
export NCCL_MIN_NCHANNELS=4
export NCCL_MIN_CTAS=16
export NCCL_MAX_CTAS=32
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_SHARP_GROUP_SIZE_THRESH=2
export NCCL_WORK_FIFO_DEPTH=1048576
export NCCL_CFG_PATH="/workspace/llm/conf/nccl/custom_communicator_cta.yaml"

# NeMo / misc                     
export NEMO_MANUAL_GC_IN_VALIDATION=0
export NEMO_LOG_TRAIN_LOSS=1
export TOKENIZERS_PARALLELISM=False
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export TQDM_DISABLE=True
export TP_PP_DP_MAPPING=True
export BINDCMD="bindpcie --cpu=node"
export EXTRA_ARGS=""
