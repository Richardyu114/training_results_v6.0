source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=32
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=8
export INTERLEAVED_PIPELINE=8
export CONTEXT_PARALLEL=2

export MICRO_BATCH_SIZE=1

export TP_COMM_OVERLAP=True
export ASYM_PP_EMBED=True
export ASYM_PP_LOSS=True

export MAX_STEPS=450

export DGXNNODES=512
export DGXNGPU=4
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=65
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export NCCL_NVLS_ENABLE=0
export UB_SKIPMC=1
export QKV_DGRAD_SM=16
export QKV_WGRAD_SM=16
export FC1_DGRAD_SM=16
export FC1_WGRAD_SM=16
