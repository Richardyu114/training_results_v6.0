#!/bin/bash

# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


if [[ -z "${SEED}" ]]; then
    echo "SEED is not set!" >&2
    exit 1
fi

mkdir -p /results

cd /workspace/code

TORCHRUN_RDZV_ARGS=()
if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
    TORCHRUN_RDZV_ARGS=(
        --rdzv_backend=c10d
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}
    )
fi

echo "============================================"
echo "MLPerf GPT-OSS-20B Training"
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

# Distributed-failure visibility (ported from llama2_sft FIXES.md):
#   * ${LOCAL_RANKS} expands (when set by run.sub) to
#       --redirects 3 --tee 3 --log-dir /results/torchelastic_logs
#     This makes torchelastic mirror EVERY worker's stdout/stderr to a
#     per-rank file under /results/torchelastic_logs/<run-id>/attempt_0/
#     <local-rank>/{stdout,stderr}.log so a failing rank's traceback is
#     preserved on the host bind mount even after the container exits.
#   * The trailing process-substitution stderr capture preserves anything
#     that torchrun (and the Python child before _log_suppression.install
#     fires) writes to FD 2, in case the in-Python FD-2 redirect ever
#     re-engages despite MLPERF_VERBOSE_LOGS=1.
# shellcheck disable=SC2206  # we explicitly want word-splitting on $LOCAL_RANKS
LOCAL_RANKS_ARGS=( ${LOCAL_RANKS:-} )

mkdir -p "/results/torchelastic_logs"

# Launch distributed training
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    "${TORCHRUN_RDZV_ARGS[@]}" \
    "${LOCAL_RANKS_ARGS[@]}" \
    src/train.py \
    2> >(tee -a "/results/train.stderr.node${NODE_RANK:-0}.log" >&2)

ret_code=$?

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result
result=$(( end - start ))
result_name="GPT_OSS_20B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
