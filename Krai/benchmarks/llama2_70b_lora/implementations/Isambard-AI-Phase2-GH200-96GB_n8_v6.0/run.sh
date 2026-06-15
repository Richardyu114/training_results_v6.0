#!/bin/bash

set -x

export SLURM_MPI_TYPE=pmi2
export MODEL_SIZE="8b"

export CONT="docker.io/krai4ai/mlperf-nvidia:llama2_70b_lora-pyt"
export DATADIR="/lus/lfs1aip2/projects/public/u5bz/krai/gov_report"
export MODEL="/lus/lfs1aip2/projects/public/u5bz/krai/model"
export LOGDIR="/lus/lfs1aip2/projects/public/u5bz/krai/output_logdir"
export CFGDIR="./"
export TRITON_CACHE_DIR="/lus/lfs1aip2/projects/public/u5bz/krai/triton_cache"

GPU_MODEL="${GPU_MODEL:-gh200}"
NN="${NN:-1}"
TP="${TP:-4}"
PP="${PP:-1}"
CP="${CP:-1}"
export CONFIG_NAME="config_${GPU_MODEL}_n${NN}_tp${TP}pp${PP}cp${CP}"
source "${CFGDIR}/${CONFIG_NAME}.sh"

export WALLTIME_RUNANDTIME=$(($BASE_TIME*$CP/$DGXNNODES)) #120
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

NE="${NE:-1}"
for ((i=1; i<=NE; i++)); do
    export NSBATCH=$i
    sbatch --job-name=${MODEL_SIZE} --exclusive \
	--time=${WALLTIME} \
  	-N ${DGXNNODES} --gpus-per-node=${DGXNGPU} \
  	--cpus-per-task ${CPUS_PER_TASK} --mem 0 \
  	--output="${LOGDIR}/slurm_logs/slurm-%j.out" \
  	run.sub
done

set +x

exit 0
