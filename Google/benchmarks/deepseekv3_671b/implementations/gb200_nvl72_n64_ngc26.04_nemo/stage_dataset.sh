#!/bin/bash
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --job-name=stage_data
#SBATCH --output=stage_data_%j.out

# Dataset setup.
# We used the script below to rsync from a shared source path to every
# compute node's local SSD in parallel.
#
# Usage: sbatch --nodelist=<your-64-nodes> stage_data.sh
#        Override SRC_DATA and DST_DATA via env or edit below.
# Skip staging if you point DATADIR at a shared FS.

SRC_DATA=${SRC_DATA:-/home/${USER}/dataset/8b}
DST_DATA=${DST_DATA:-/mnt/localssd/dataset/8b}

srun --ntasks-per-node=1 bash -c "
  set -euo pipefail
  hostname
  mkdir -p ${DST_DATA}
  time rsync -a --partial --inplace --no-compress ${SRC_DATA}/ ${DST_DATA}/
  echo '--- verify ---'
  du -sh ${DST_DATA}
  find ${DST_DATA} -type f | wc -l
"
