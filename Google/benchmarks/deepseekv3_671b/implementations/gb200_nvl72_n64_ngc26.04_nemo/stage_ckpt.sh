#!/bin/bash
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=stage_ckpt
#SBATCH --output=stage_ckpt_%j.out

# Checkpoint setup.
# Stages the Megatron-Bridge converted checkpoint (~1.3 TB) from a shared
# source path to every compute node's local SSD in parallel.
#
# Usage: sbatch --nodelist=<your-64-nodes> stage_ckpt.sh
#        Override SRC_CKPT and DST_CKPT via env or edit below.
# Skip staging if you point LOAD_CHECKPOINTS_DIR at a shared FS.

SRC_CKPT=${SRC_CKPT:-/home/${USER}/mbridge_ckpt}
DST_CKPT=${DST_CKPT:-/mnt/localssd/mbridge_ckpt}

srun --ntasks-per-node=1 bash -c "
  set -euo pipefail
  hostname
  mkdir -p ${DST_CKPT}
  time rsync -a --partial --inplace --no-compress ${SRC_CKPT}/ ${DST_CKPT}/
  echo '--- verify ---'
  du -sh ${DST_CKPT}
  find ${DST_CKPT} -type f | wc -l
"
