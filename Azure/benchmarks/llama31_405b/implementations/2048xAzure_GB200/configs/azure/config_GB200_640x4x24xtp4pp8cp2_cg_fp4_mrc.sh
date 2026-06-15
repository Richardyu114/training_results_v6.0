source $(dirname ${BASH_SOURCE[0]})/config_mrc_env.sh
source $(dirname ${BASH_SOURCE[0]})/../nvidia/config_GB200_640x4x24xtp4pp8cp2_cg_fp4.sh

# ==============================================================================
# MRC-specific overrides
# ==============================================================================
export WALLTIME_RUNANDTIME=340

_V6_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LOGROOT="${_V6_DIR}/experiments/logs_${DGXNNODES}_${MINIBS}"
export LOGDIR="${LOGROOT}/$(date +%y%m%d%H%M%S%N)"
mkdir -p "${LOGDIR}"

export DATADIR=/mnt/resource_nvme/mlperf/data/
export CONT=/mnt/resource_nvme/mlperf/nvdlfwea+mlperftv60+llama31_405b-arm+20260416.sqsh
export LOAD_CHECKPOINT="/load_checkpoints/405b"
export LOAD_CHECKPOINTS_PATH=/mnt/resource_nvme/mlperf/nemo-formatted-hf-checkpoint

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET

export BUCKET_SIZE=134217728

# Mount only the modified run_and_time.sh into the container (keeps original /workspace/llm intact)
export EXTRA_MOUNTS="${EXTRA_MOUNTS},${_V6_DIR}/run_and_time.sh:/workspace/llm/run_and_time.sh:ro"
