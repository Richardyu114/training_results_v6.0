source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh


export MINIBS=1
export MICRO_BATCH_SIZE=1
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=2
export EXPERT_PARALLEL=4
# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=1

export CUDA_GRAPH_IMPLEMENTATION="transformer_engine"
export CUDA_GRAPH_SCOPE="attn,moe_router,moe_preprocess"

export LR=0.0004
export VAL_CHECK_INTERVAL=341
export LR_WARMUP_STEPS=256

export DGXNNODES=18
export DGXNGPU=4
export SEGMENT=18
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=100
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
