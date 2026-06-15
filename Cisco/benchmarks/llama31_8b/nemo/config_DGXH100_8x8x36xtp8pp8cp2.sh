source $(dirname ${BASH_SOURCE[0]})/config_common.sh

export MICRO_BATCH_SIZE=1
export MINIBS=1

export LR=0.0008
export MIN_LR=0.00008
export TENSOR_MODEL_PARALLEL=2
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1
export SEQ_PARALLEL=True
export TP_COMM_OVERLAP=True

export DGXNNODES=8
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=420
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export WARMUP_STEPS=192
export LR_WARMUP_STEPS=$((MAX_STEPS / 10))
export MAX_STEPS_FOR_LR_SCHED=1199808
export OPT_LR_DECAY_STEPS=${MAX_STEPS}
export VAL_CHECK_INTERVAL=384

export BUCKET_CAP_MB=125
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
