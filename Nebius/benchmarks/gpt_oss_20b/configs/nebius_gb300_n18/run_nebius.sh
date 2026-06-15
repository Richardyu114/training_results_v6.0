#!/bin/bash
#
#
export MODELNAME="gpt_oss_20b"
export DATADATE="20260507"

export CONT="/mnt/slurm/mlperf/images/arm/${MODELNAME}_${DATADATE}.sqsh"
export DATADIR="/mnt/slurm/mlperf/datasets/llama31_8b"

export GLOO_SOCKET_IFNAME=eth0
export NEXP=10
export NCCL_TEST=0
export CHECK_COMPLIANCE=1

CONFIGS=(
        # config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh
        config_GB300_18x4x1xtp1pp1cp2ep4_mxfp8.sh
)

for CONFIG in "${CONFIGS[@]}"; do
        (
                VERSION="${CONT##*_}"
                VERSION="${VERSION%.*}"
                export LOGDIR="/mnt/slurm/mlperf/results/${MODELNAME}/${VERSION}/${CONFIG%.sh}"

                source "${CONFIG}"
                JOB_NAME="${MODELNAME}_${CONFIG%.sh}"
                echo "Submitting job for ${JOB_NAME} (nodes=${DGXNNODES}, gpus=${DGXNGPU}, walltime=${WALLTIME})"
                sbatch --job-name="${JOB_NAME}" \
                        --hold \
                        --exclusive \
                        --gpus-per-node="${DGXNGPU}" \
                        -N "${DGXNNODES}" \
                        --time="${WALLTIME}" \
                        run.sub
        )
done
