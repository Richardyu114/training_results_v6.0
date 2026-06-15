#!/bin/bash

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

echo "nnodes: $DGXNNODES"
echo "nproc_per_node: $DGXNGPU"

if [[ "${AMD_SMI_DEBUG:-0}" == "1" ]]; then amd-smi; fi

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE

echo "RANK: $RANK"
echo "LOCAL_RANK $LOCAL_RANK"
echo "WORLD_SIZE $WORLD_SIZE"
echo "LOCAL_WORLD_SIZE $LOCAL_WORLD_SIZE"

python3 src/train.py
ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( end - start ))
result_name="LLM_FINETUNING"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

exit 0
