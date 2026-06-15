source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh


export MINIBS=2
export MICRO_BATCH_SIZE=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=1
# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=0

# Knobs for CG + CuteDSL
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDA_GRAPH_WARMUP_STEPS=5
export REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG=True
export FP8_PARAM_GATHER=True
export MOE_EXPERT_RANK_CAPACITY_FACTOR=1.2

export LR=0.0004
export VAL_CHECK_INTERVAL=768
export LR_WARMUP_STEPS=128

export DGXNNODES=1
export DGXNGPU=8

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=2000
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
