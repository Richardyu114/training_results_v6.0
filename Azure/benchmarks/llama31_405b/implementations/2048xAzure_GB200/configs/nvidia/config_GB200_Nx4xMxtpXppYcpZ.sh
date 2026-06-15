# GB200 config that allows free adjustments
# Default config: MINIBS=63, TP4PP9CP2VP7 9x4 GPUs MBS=1

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh$
###
: '
Guidelines on the following knobs: 
The following conditions need to be met for experiments to run: 
- (GBS // PP) % VP == 0, where GBS = (MINIBS * NNODES * NGPUS) // (TP * PP * CP)
- OVERWRITTEN_NUM_LAYERS % (PP * VP) == 0

Recommendation: 
- TP * PP * CP <= NNODES * NGPUS
    - this is very important, otherwise will trigger division by 0 error. 
- MINIBS % VP == 0
- (OVERWRITTEN_NUM_LAYERS // PP) % VP == 0

Example: 
MINIBS=12 TP=4 PP=2 VP=6 CP=2 OVERWRITTEN_NUM_LAYERS=12 NNODES=4 on GB200: 
- 4 * 2 * 2 <= 4 * 4
- 12 % 6 = 0
- (12 // 2 = 6) % 6 = 0

Example: 
MINIBS=36 TP=4 PP=2 VP=2 CP=2 OVERWRITTEN_NUM_LAYERS=32 on 4xGB200: 
- 4 * 2 * 2 <= 4 * 4
- 36 % 2 = 0
- (32 // 2 = 16) % 2 == 0

To use this config: export all of the above environment variable 
and then launch experiment with this config file using launch-mlperf-benchmark.sh
'
###

export MINIBS="${MINIBS:=63}"
export TENSOR_MODEL_PARALLEL="${TP:=4}"
export PIPELINE_MODEL_PARALLEL="${PP:=9}"
export INTERLEAVED_PIPELINE="${VP:=7}"
export CONTEXT_PARALLEL="${CP:=2}"

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=1

export OVERWRITTEN_NUM_LAYERS=${OVERWRITTEN_NUM_LAYERS:=126} # defaults to 126 layers, which is the original number of layers
# if the model shape does not match then do not load checkpoint
if [[ ! $OVERWRITTEN_NUM_LAYERS -eq 126 ]]; then export LOAD_CHECKPOINT=""; fi

export LIMIT_TRAIN_BATCHES=10
export MAX_STEPS=10
export LIMIT_VAL_BATCHES=0
export VAL_CHECK_INTERVAL=15
export FORCE_SUCCESS_STATUS=1

export DGXNNODES="${DGXNNODES:=9}"
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=60
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))
