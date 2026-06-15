source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=1152
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=18
export INTERLEAVED_PIPELINE=7
export CONTEXT_PARALLEL=1

export MICRO_BATCH_SIZE=1

#export TP_COMM_OVERLAP=True
export TP_COMM_OVERLAP=False

export MAX_STEPS=600
export VAL_CHECK_INTERVAL=50

export LOAD_CHECKPOINTS_PATH="/wekafs/crickett/llama3.1/checkpoint/new-ckpt"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINTS_PATH}/405b"

export DGXNNODES=18
export DGXNGPU=4
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=800
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

