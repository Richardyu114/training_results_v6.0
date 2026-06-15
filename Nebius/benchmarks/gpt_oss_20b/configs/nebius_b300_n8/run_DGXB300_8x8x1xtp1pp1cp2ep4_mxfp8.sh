#!/bin/bash

: "${WORKER_IDS:=0}"
: "${NEXP:=1}"
: "${GPU_ARCH:=b}"
: "${CPUS_PER_TASK:=24}"

echo "WORKER_IDS=${WORKER_IDS}"
echo "NEXP=${NEXP}"

export SLURM_MPI_TYPE=pmi2
export UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export PMIX_GDS_MODULE=^ds12
export PMIX_MCA_gds=^ds12
export NCCL_NET_PLUGIN=none

export WORKDIR="/mnt/data/training_v6.0/training_results_v6.0_Nebius-benchmarks.gpt_oss_20b.20260508/benchmarks/gpt_oss_20b"
export CONT="/mnt/data/training_v6.0/images/gpt_oss_20b_20260508.sqsh"
export DATADIR="/mnt/data/training_v6.0/datasets/llama31_8b"
export CFGDIR="./"
export LOGDIR="/mnt/data/training_v6.0/runs/b300_n8/gpt_oss_20b/logs.20260509"

source "${CFGDIR}/config_DGXB300_8x8x1xtp1pp1cp2ep4_mxfp8.sh"
cd ${WORKDIR}
sbatch --job-name=${MODEL_SIZE} --exclusive \
  -N ${DGXNNODES} --gpus-per-node=${DGXNGPU} \
  --cpus-per-task ${CPUS_PER_TASK} --mem 0 \
  --nodelist="worker-[${WORKER_IDS}]" \
  --output="${LOGDIR}/slurm-%j.out" \
  run.sub

date

exit 0
