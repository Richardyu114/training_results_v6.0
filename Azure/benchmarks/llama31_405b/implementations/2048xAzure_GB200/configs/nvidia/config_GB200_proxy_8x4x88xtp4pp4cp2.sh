source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh

export MINIBS=88
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=4
export INTERLEAVED_PIPELINE=8
export CONTEXT_PARALLEL=2

export TP_COMM_OVERLAP=True
export MICRO_BATCH_SIZE=1

export OVERWRITTEN_NUM_LAYERS=62
# if the model shape does not match then do not load checkpoint
if [[ ! $OVERWRITTEN_NUM_LAYERS -eq 126 ]]; then export LOAD_CHECKPOINT=""; fi

export MAX_STEPS=20
export LIMIT_VAL_BATCHES=2
export VAL_CHECK_INTERVAL=6
export FORCE_SUCCESS_STATUS=1
#export MOCK_DATASET="False"
export MOCK_DATASET="True"

export DGXNNODES=8
export DGXNGPU=4
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=45
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export ASYM_PP_EMBED=True
export ASYM_PP_LOSS=True
