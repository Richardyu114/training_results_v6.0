#!/bin/bash
# run_and_time_4node.sh — multi-node trial runner.
#
# Diverges from `run_and_time.sh` by NOT redirecting torchrun's stderr to
# /dev/null, so NCCL/Gloo bootstrap failures, RDMA registration errors, and
# rank-level python tracebacks land in the trial log. _log_suppression.py
# (controlled by MLPERF_VERBOSE_LOGS) still handles in-process noise.

set -e

mkdir -p /results
cd /workspace/code

echo "============================================"
echo "MLPerf LLama3.1 8B Training (4-node)"
echo "============================================"
echo "Config: ${EXP}"
echo "Data:   ${DATA_PATH}"
echo "GPUs:   ${GPUS_PER_NODE}"
echo "Nodes:  ${NNODES}"
echo "Rank:   ${NODE_RANK}/${NNODES}  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py

ret_code=$?

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

result=$(( end - start ))
result_name="LLAMA3.1_8B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
