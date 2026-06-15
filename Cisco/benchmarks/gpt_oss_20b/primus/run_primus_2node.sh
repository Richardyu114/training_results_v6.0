#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "${SCRIPT_DIR}"
export PRIMUS_SCRIPT_DIR="${SCRIPT_DIR}"

export CONFIG_NAME="${CONFIG_NAME:-config_MI350X_2x8x1_tp1pp1ep1_gbs64.sh}"

: "${CONT:?CONT not set (path to .sqsh image or dockerd://... reference)}"
: "${DATADIR:?DATADIR not set}"
: "${MODELDIR:?MODELDIR not set}"
export CONT DATADIR MODELDIR
export LOGDIR="${LOGDIR:-${SCRIPT_DIR}/results/MI350X_2x8x1}"
export NEXP="${NEXP:-1}"

mkdir -p "${LOGDIR}"

sbatch_args=(--nodes=2)
[[ -n "${SLURM_NODELIST:-}"  ]] && sbatch_args+=(--nodelist="${SLURM_NODELIST}")
[[ -n "${SLURM_PARTITION:-}" ]] && sbatch_args+=(--partition="${SLURM_PARTITION}")
[[ -n "${SLURM_TIME:-}"      ]] && sbatch_args+=(--time="${SLURM_TIME}")

sbatch "${sbatch_args[@]}" run.sub
