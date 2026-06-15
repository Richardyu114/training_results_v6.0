#!/usr/bin/env bash
# Launch NEXP=10 timed runs of LLaMA2-70B LoRA on 2x HGX B300.
# Wraps the NVIDIA reference run.sub which lives inside the container/workdir.
set -euo pipefail

: "${WORKDIR:=$HOME/training60/llama2-70b-work-v2}"
: "${DATAROOT:=$HOME/training60}"
: "${SQSH_DIR:=$HOME/sqsh}"

cd "${WORKDIR}"

# Pull in the implementation config from the in-container NVIDIA reference.
source config_HGXB300_2x8x1xtp1pp1cp1.sh

export SLURM_MPI_TYPE=pmi2
export CONT="${SQSH_DIR}/llama2_70b_lora_20260507.sqsh"
export DATADIR="${DATAROOT}/data/gov_report"
export MODEL="${DATAROOT}/model-v2"
export LOGDIR="${WORKDIR}/results"
export NEXP=10

sbatch -N "${DGXNNODES}" \
  --time=$((10 * WALLTIME)) \
  --job-name=mlperf_llama2_70b_nexp10 \
  --partition=mlperf \
  run.sub

squeue
