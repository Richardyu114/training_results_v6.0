source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh

export MINIBS=8
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=2

export MICRO_BATCH_SIZE=1

export OVERWRITTEN_NUM_LAYERS=2
export LOAD_CHECKPOINT=
export OVERLAP_P2P_COMM=False
export OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=0

# APILog - we don't need these for our runs
export CHECK_COMPLIANCE=0
export NCCL_TEST=0
export APILOG_PRECISION="fp8" # (or "fp32")
export BENCHMARK="llama31_405b"
export FRAMEWORK="pytorch"
export BATCHSIZE=${MICRO_BATCH_SIZE}

# Set clocks
if [[ "${SET_GPU_CLK:-1}" == "1" ]]; then
    export MAX_GPU_CLK=1815
fi

export TP_COMM_OVERLAP=True

export DGXNNODES=2
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=15
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export LIMIT_TRAIN_BATCHES=10
export LIMIT_VAL_BATCHES=2
export VAL_CHECK_INTERVAL=10

export MAX_STEPS=10
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))
