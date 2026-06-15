source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

# DM: Bumping up to 6 since 2 has low mem ut
export MINIBS=6
export MICRO_BATCH_SIZE=6
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export LR=0.0004
export WARMUP_STEPS=16
export VAL_CHECK_INTERVAL=768

export DGXNNODES=18
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=360
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# Persist the NPY dataset index across runs so rank 0 doesn't rebuild it from
# scratch every time (which takes >600s under 18-node filesystem pressure and
# causes the other 71 ranks to time out waiting at the on_data_init_start barrier).
# Pre-build: source this config, override DGXNNODES=2, and run a 2-node job first.
export NPY_INDEX_DIR=/sharedfs/dj/mlperf_train_v60/llama31_8b_stuff/npy_cache
export CLEANUP_NPY_INDEX_DIR=0

