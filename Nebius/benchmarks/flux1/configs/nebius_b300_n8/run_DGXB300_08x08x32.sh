#!/bin/bash

: "${WORKER_IDS:=0,1,2,3,4,5,6,7}"
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
export JET=0

export WORKDIR="/mnt/data/training_v6.0/training_results_v6.0_Nebius-benchmarks.flux1.20260511/benchmarks/flux1"
export CONT="/mnt/data/training_v6.0/images/flux1_20260511.sqsh"
export DATADIR="/mnt/data/training_v6.0/datasets/flux1"
export CFGDIR="./"
export LOGDIR="/mnt/data/training_v6.0/runs/b300_n8/flux1/logs.20260512"
export DATAROOT="${DATADIR}/energon"

source "${CFGDIR}/config_DGXB300_08x08x32.sh"
cd ${WORKDIR}
sbatch --job-name=flux1 --exclusive \
  -N ${DGXNNODES} --gpus-per-node=${DGXNGPU} \
  --cpus-per-task ${CPUS_PER_TASK} --mem 0 \
  --nodelist="worker-[${WORKER_IDS}]" \
  --output="${LOGDIR}/slurm-%j.out" \
  run.sub

date

exit 0
