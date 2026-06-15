#!/bin/bash

export CONT="/mnt/slurm/mlperf/images/arm/llama31_8b_20260507.sqsh"
export DATADIR="/mnt/slurm/mlperf/datasets/llama31_8b/"

export GLOO_SOCKET_IFNAME=eth0

export NEXP=10
export NCCL_TEST=0

CONFIGS=(
	# config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh
	config_GB300_18x4x1xtp1pp1cp1_8b_fp4.sh
)

for CONFIG in "${CONFIGS[@]}"; do
	(
		VERSION="${CONT##*_}"
		VERSION="${VERSION%.*}"
		export LOGDIR="/mnt/slurm/mlperf/results/llama31_8b/${VERSION}/${CONFIG%.sh}"

		source "${CONFIG}"
		JOB_NAME="llama31_8b_${CONFIG%.sh}"
		echo "Submitting job for ${JOB_NAME} (nodes=${DGXNNODES}, gpus=${DGXNGPU}, walltime=${WALLTIME})"
		sbatch --job-name="${JOB_NAME}" \
			--exclusive \
			--gpus-per-node="${DGXNGPU}" \
			-N "${DGXNNODES}" \
			--time="${WALLTIME}" \
			run.sub
	)
done
