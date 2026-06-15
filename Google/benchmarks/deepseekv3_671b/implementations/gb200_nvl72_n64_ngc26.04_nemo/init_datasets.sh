#!/bin/bash
# Dataset and checkpoint setup.
#
# Follow upstream MLPerf Training v6.0 DeepSeek-V3 671B README for:
#   - Dataset download (bash data_scripts/download.sh)
#   - Checkpoint download (MLCommons R2 downloader)
#   - Checkpoint conversion (HF -> Megatron-Bridge)
#
# For our submission we additionally staged both onto per-node local SSD
# (/mnt/localssd) via the following Slurm scripts:
#   stage_dataset.sh   -- rsyncs dataset to /mnt/localssd/dataset/8b
#   stage_ckpt.sh      -- rsyncs converted checkpoint to /mnt/localssd/mbridge_ckpt
#
# Skip staging if you point DATADIR / LOAD_CHECKPOINTS_DIR at a shared FS.

set -euo pipefail

echo "Dataset/checkpoint staging is optional. See:"
echo "  - README.md section 2 (Prepare dataset and checkpoint)"
echo "  - stage_dataset.sh"
echo "  - stage_ckpt.sh"
