#!/bin/bash
# Launch gpt_oss_20b on B200 8x8. Optionally chain after another job by
# setting DEPENDS_ON=<jobid>. Run from the dir holding run.sub + config_*.sh:
#   cd "$(dirname "$0")" && [DEPENDS_ON=<jobid>] bash submit_20b.sh
set -euo pipefail

export NEXP=10
export DATADIR=/mnt/data/gpt-oss-20b/dataset
export LOGDIR=$PWD/logs
mkdir -p "$LOGDIR"

source config_DGXB200_8x8x1xtp1pp1cp2ep4_mxfp8.sh
echo "[gpt_oss_20b] NEXP=$NEXP DGXNNODES=$DGXNNODES WALLTIME=$WALLTIME"

DEP_FLAG=""
if [[ -n "${DEPENDS_ON:-}" ]]; then
  DEP_FLAG="--dependency=afterany:${DEPENDS_ON}"
fi

CONT=/mnt/data/images/gpt_oss_20b-amd-20260508.sqsh \
  sbatch --parsable --export=ALL \
         -p hpc-high -N 8 --gres=gpu:b200:8 \
         -t $((WALLTIME+30)) \
         ${DEP_FLAG} \
         run.sub
