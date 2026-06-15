#!/bin/bash
# Submit 2-node Slurm job: full environment from run_env.sh, then run.sub (Pyxis + srun only).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/run_env.sh"

sbatch \
    --job-name=llama2-70b-lora-2node \
    --nodes=2 \
    --exclusive \
    --export=ALL \
    "${SCRIPT_DIR}/run.sub"
