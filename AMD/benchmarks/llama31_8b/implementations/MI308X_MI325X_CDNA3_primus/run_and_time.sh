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

echo "============================================"
echo "MLPerf LLama3.1 8B Training"
echo "============================================"
echo "Config: ${EXP}"
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

eval torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    src/train.py ${_stderr_redirect}

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
