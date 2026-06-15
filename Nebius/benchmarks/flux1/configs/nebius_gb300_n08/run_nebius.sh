#!/bin/bash
#
#
export MODELNAME="flux1"
export DATADATE="20260511"

export CONT="/mnt/slurm/mlperf/images/arm/${MODELNAME}_${DATADATE}.sqsh"
export DATAROOT="/mnt/slurm/mlperf/datasets/${MODELNAME}/energon"

export GLOO_SOCKET_IFNAME=eth0
export NEXP=10
export NCCL_TEST=0
export CHECK_COMPLIANCE=1

export PMIX_GDS_MODULE=^ds12
export PMIX_MCA_gds=^ds12
export BINDCMD="bindpcie --cpu=node"
#export SLURM_CPUS_PER_TASK=28

CONFIGS=(
        config_GB300_08x04x32.sh
        # config_GB300_18x04x32.sh
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
