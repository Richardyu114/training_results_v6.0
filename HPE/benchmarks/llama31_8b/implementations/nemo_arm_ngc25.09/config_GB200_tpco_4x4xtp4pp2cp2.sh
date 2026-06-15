source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh

export COMM_GEMM_OVERLAP_TEST=1
export RUN_ONLY_COMM_GEMM_OVLP=0

export MINIBS=32
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=2
export INTERLEAVED_PIPELINE=2
export CONTEXT_PARALLEL=2

export TP_COMM_OVERLAP=True
export MICRO_BATCH_SIZE=1

export OVERWRITTEN_NUM_LAYERS=4
# if the model shape does not match then do not load checkpoint
if [[ ! $OVERWRITTEN_NUM_LAYERS -eq 126 ]]; then export LOAD_CHECKPOINT=""; fi

export LIMIT_TRAIN_BATCHES=120
export MAX_STEPS=120
export LIMIT_VAL_BATCHES=5
export VAL_CHECK_INTERVAL=20
export FORCE_SUCCESS_STATUS=1
export MOCK_DATASET="False"

# Binding
export BINDCMD="bindpcie --cpu=node"

export DGXNNODES=4
export DGXNGPU=4
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=15
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
