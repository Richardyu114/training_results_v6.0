#!/bin/bash
# One-time setup for gb200_nvl72_n64_ngc26.04_nemo

set -euo pipefail

# 1. Build container and convert to squashfs
DOCKER_TAG=${DOCKER_TAG:-deepseekv3_671b:gb200-mlperf}
SQSH_PATH=${SQSH_PATH:-${HOME}/images/deepseekv3_671b.sqsh}

DOCKER_BUILDKIT=1 docker build \
  --build-arg GIT_COMMIT_ID=$(git rev-parse HEAD) \
  -t ${DOCKER_TAG} \
  ../gb200_megatron_bridge/

mkdir -p $(dirname ${SQSH_PATH})
enroot import -o ${SQSH_PATH} dockerd://${DOCKER_TAG}

# 2. Verify Slurm + Pyxis available
command -v sbatch >/dev/null || { echo "Slurm not found"; exit 1; }

# 3. Verify NVLink fabric topology — 4 cliques expected, 16 nodes per clique for EP=32
echo "Run the following to verify clique distribution before launching:"
echo "  srun -N64 -l --ntasks-per-node=1 nvidia-smi --query-gpu=fabric.clique_id --format=csv,noheader | sort | uniq -c"

echo "Setup complete. Container image: ${SQSH_PATH}"
echo "Next steps:"
echo "      sbatch stage_dataset.sh and stage_ckpt.sh (or skip if using shared FS),"
echo "      then source config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh && sbatch run.sub"
