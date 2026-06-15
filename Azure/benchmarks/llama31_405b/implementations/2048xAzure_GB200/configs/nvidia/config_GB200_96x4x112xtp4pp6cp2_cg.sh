source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=112
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=6
export INTERLEAVED_PIPELINE=7
export CONTEXT_PARALLEL=2

export MICRO_BATCH_SIZE=1

export DEFER_EMBEDDING_WGRAD_COMPUTE=False

export TP_COMM_OVERLAP=True

export MAX_STEPS=450

export DGXNNODES=96
export DGXNGPU=4
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=220
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
