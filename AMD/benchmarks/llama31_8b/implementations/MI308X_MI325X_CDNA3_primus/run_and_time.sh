#!/bin/bash

set -e

# Create results directory
mkdir -p /results

cd /workspace/code

# Under multi-node SLURM (run_with_docker_slurm.sh), inherit rendezvous + node
# sizing from SLURM env so we can scale to N nodes without editing the config
# file. Single-node SLURM jobs (NNODES=1) fall through to the config defaults
# so torchrun doesn't try to do c10d rdzv against MASTER_ADDR=localhost.
if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
fi

# Keep MLLOG precision metadata aligned with the final experiment override.
# Platform configs default to FP8, but launchers may replace EXP with the BF16
# yaml after sourcing the config. The yaml remains the runtime source of truth.
: "${EXP:?EXP not set}"
if [[ ! -r "${EXP}" ]]; then
    echo "ERROR: experiment config is not readable: ${EXP}" >&2
    exit 2
fi
if grep -Eq '^[[:space:]]*fp8:[[:space:]]*(hybrid|e4m3)([[:space:]]|#|$)' "${EXP}"; then
    _training_precision=fp8
    export FP8=true
elif [[ "${EXP##*/}" == *bf16*.yaml ]]; then
    _training_precision=bf16
    export FP8=false
else
    echo "ERROR: cannot determine training precision from EXP=${EXP}" >&2
    exit 2
fi
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR="${_training_precision}"

echo "============================================"
echo "MLPerf LLama3.1 8B Training"
echo "============================================"
echo "Config: ${EXP}"
echo "Precision: ${_training_precision}"
echo "Data:   ${DATA_PATH}"
echo "GPUs:   ${GPUS_PER_NODE}"
echo "Nodes:  ${NNODES}"
echo "Rank:   ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Launch distributed training. In the default (quiet) mode stderr is dropped and
# _log_suppression.py filters stdout, matching the original submission. To see the per-iteration
# training log (loss + throughput), run with MLPERF_VERBOSE_LOGS=1: that both skips the in-process
# suppression (see src/_log_suppression.py) and keeps stderr here, where loguru emits the training
# line (requires stderr_sink_level=INFO, the default in the yaml).
if [[ "${MLPERF_VERBOSE_LOGS:-0}" == "1" ]]; then
    _stderr_redirect=""   # keep stderr so the loss/throughput line is visible
else
    _stderr_redirect="2>/dev/null"
fi

# Rendezvous mode. For multi-node we MUST use the static backend (--master_addr/--master_port
# + --node_rank): rank0 binds+listens on MASTER_ADDR directly, which works with a plain IP.
# The c10d backend instead picks the master by matching a node's `hostname` output against the
# rdzv_endpoint host; with an IP endpoint no node matches, so nobody starts the store and every
# rank times out on TCPStore connect. Mixing both styles also makes c10d silently ignore
# master_addr/port. Single-node keeps the original c10d line (localhost matches, works fine).
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
result_name="LLAMA3.1_8B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
