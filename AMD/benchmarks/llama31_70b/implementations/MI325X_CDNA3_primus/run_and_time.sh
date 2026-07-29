#!/bin/bash

set -e

# Create results directory
mkdir -p /results

cd /workspace/code

# Under multi-node SLURM, inherit rendezvous + node sizing from SLURM env so we can scale
# without editing the config file. Single-node SLURM jobs (NNODES=1) fall through to the
# config defaults so torchrun doesn't try c10d rdzv against MASTER_ADDR=localhost.
if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
fi

# Derive the precision label from the experiment yaml, which is the runtime source of
# truth. An `fp8:` block wins; otherwise fall back to the fp16/bf16 marker in the filename.
: "${EXP:?EXP not set}"
if [[ ! -r "${EXP}" ]]; then
    echo "ERROR: experiment config is not readable: ${EXP}" >&2
    exit 2
fi
if grep -Eq '^[[:space:]]*fp8:[[:space:]]*(hybrid|e4m3)([[:space:]]|#|$)' "${EXP}"; then
    _training_precision=fp8
    export FP8=true
elif [[ "${EXP##*/}" == *fp16*.yaml ]]; then
    _training_precision=fp16
    export FP8=false
    # Guard against a silent bf16 run: Primus' trainer_base defaults bf16 to true, so an
    # fp16 yaml that forgets `bf16: false` would train in bf16 while claiming fp16.
    if ! grep -Eq '^[[:space:]]*fp16:[[:space:]]*true([[:space:]]|#|$)' "${EXP}"; then
        echo "ERROR: ${EXP} is named fp16 but does not set 'fp16: true'" >&2
        exit 2
    fi
    if ! grep -Eq '^[[:space:]]*bf16:[[:space:]]*false([[:space:]]|#|$)' "${EXP}"; then
        echo "ERROR: ${EXP} sets fp16 but does not set 'bf16: false'; Primus defaults" \
             "bf16 to true, so training would run in bf16" >&2
        exit 2
    fi
elif [[ "${EXP##*/}" == *bf16*.yaml ]]; then
    _training_precision=bf16
    export FP8=false
else
    echo "ERROR: cannot determine training precision from EXP=${EXP}" >&2
    exit 2
fi
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR="${_training_precision}"

echo "============================================"
echo "LLama3.1 70B Training"
echo "============================================"
echo "Config:    ${EXP}"
echo "Precision: ${_training_precision}"
echo "Data:      ${DATA_PATH}"
echo "GPUs:      ${GPUS_PER_NODE}"
echo "Nodes:     ${NNODES}"
echo "Rank:      ${NODE_RANK}"
echo "Master:    ${MASTER_ADDR}:${MASTER_PORT}"
echo "TP x PP:   ${PRIMUS_TENSOR_PARALLEL_SIZE:-?} x ${PRIMUS_PIPELINE_PARALLEL_SIZE:-?}"
echo "MBS/GBS:   ${PRIMUS_MICRO_BATCH_SIZE:-?} / ${PRIMUS_GLOBAL_BATCH_SIZE:-?}"
echo "============================================"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# In the default (quiet) mode stderr is dropped and _log_suppression.py filters stdout.
# MLPERF_VERBOSE_LOGS=1 skips the in-process suppression and keeps stderr, where loguru
# emits the per-iteration loss + TFLOP/s line (requires stderr_sink_level=INFO).
if [[ "${MLPERF_VERBOSE_LOGS:-0}" == "1" ]]; then
    _stderr_redirect=""
else
    _stderr_redirect="2>/dev/null"
fi

# Rendezvous mode. For multi-node we MUST use the static backend (--master_addr/--master_port
# + --node_rank): rank0 binds+listens on MASTER_ADDR directly, which works with a plain IP.
# The c10d backend instead picks the master by matching a node's `hostname` output against the
# rdzv_endpoint host; with an IP endpoint no node matches, so nobody starts the store and every
# rank times out on TCPStore connect. Single-node keeps the c10d line (localhost matches).
if [[ "${NNODES}" -gt 1 ]]; then
    eval torchrun \
        --nproc_per_node=${GPUS_PER_NODE} \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        src/train.py ${_stderr_redirect}
else
    eval torchrun \
        --nproc_per_node=${GPUS_PER_NODE} \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        src/train.py ${_stderr_redirect}
fi

ret_code=$?

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result
result=$(( end - start ))
result_name="LLAMA3.1_70B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
