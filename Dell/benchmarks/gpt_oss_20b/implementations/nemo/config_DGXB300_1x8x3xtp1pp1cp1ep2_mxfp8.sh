source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh


export MINIBS=3
export MICRO_BATCH_SIZE=3
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=2
# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=0

export CUDA_GRAPH_IMPLEMENTATION="transformer_engine"
export CUDA_GRAPH_SCOPE="attn,moe_router,moe_preprocess"

export LR=0.0005
export VAL_CHECK_INTERVAL=512
export LR_WARMUP_STEPS=256

export DGXNNODES=1
export DGXNGPU=8

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=200
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

