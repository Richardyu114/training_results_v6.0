#!/usr/bin/env bash
# Launch NEXP=10 timed runs of LLaMA3.1 8B pretraining on 2x HGX B300.
set -euo pipefail

: "${WORKDIR:=$HOME/training60/llama3-8b-work}"
: "${DATAROOT:=$HOME/training60}"
: "${SQSH_DIR:=$HOME/sqsh}"

cd "${WORKDIR}"

source config_HGXB300_2x8x2xtp1pp1cp1_8b.sh

unset PYTHONWARNINGS
export SLURM_MPI_TYPE=pmi2
export CONT="${SQSH_DIR}/llama31_8b_20260507.sqsh"
export DATADIR="${DATAROOT}/data"
export LOGDIR="${WORKDIR}/results"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IGNORE_COLLNET_MISMATCH=1
export NEXP=10

sbatch -N "${DGXNNODES}" \
  --time=$((10 * WALLTIME)) \
  --job-name=mlperf_llama3_8b_nexp10 \
  --partition=mlperf \
  run.sub

squeue
