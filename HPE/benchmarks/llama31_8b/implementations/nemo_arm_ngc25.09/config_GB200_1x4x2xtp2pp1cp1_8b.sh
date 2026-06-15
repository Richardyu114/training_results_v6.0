source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh

export MINIBS=2
export TENSOR_MODEL_PARALLEL=2
export SEQ_PARALLEL=True

export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=True
export MICRO_BATCH_SIZE=1

export OVERLAP_GRAD_REDUCE=False
export OVERLAP_PARAM_GATHER=False

export WARMUP_STEPS=96
export VAL_CHECK_INTERVAL=192

export LR=0.0008

export DGXNNODES=1
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=30
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))