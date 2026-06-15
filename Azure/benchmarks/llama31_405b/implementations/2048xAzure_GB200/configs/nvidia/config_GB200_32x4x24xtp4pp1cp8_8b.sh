source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh

export MINIBS=24
export TENSOR_MODEL_PARALLEL=4
export SEQ_PARALLEL=True
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=8

export TP_COMM_OVERLAP=True
export MICRO_BATCH_SIZE=1

export LR=0.002
export WARMUP_STEPS=340
export VAL_CHECK_INTERVAL=128


export DGXNNODES=32
export DGXNGPU=4
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=80
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
