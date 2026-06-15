source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=1
export MICRO_BATCH_SIZE=1
export TENSOR_MODEL_PARALLEL=2
export SEQ_PARALLEL=True
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=4
export EXPERT_PARALLEL=8
# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=1

# oversized activation for CG. Override the env. var set in config_common_cg.sh
export MOE_EXPERT_RANK_CAPACITY_FACTOR=7

# Enable CuteDSL kernels
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

export NCCL_UB_DP=False

export LR=0.00052
export VAL_CHECK_INTERVAL=192
export LR_WARMUP_STEPS=32

export DGXNNODES=128
export DGXNGPU=4
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * EXPERT_PARALLEL * CONTEXT_PARALLEL) / 4 ))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=25
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

