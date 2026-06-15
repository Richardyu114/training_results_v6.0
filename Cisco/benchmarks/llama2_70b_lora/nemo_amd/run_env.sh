# Llama2-70B LoRA: all environment for Slurm submit (and paths + cluster for Docker).
# Source from run_2node.sh before sbatch. For Docker, set RUN_ENV_SKIP_CONFIG=1 first if you
# already sourced a config_*.sh (avoids loading CONFIG_FILE again).
#
_CODE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
# Real host path to this directory (Slurm copies run.sub to spool; $0 there is not the repo).
export CODE_DIR="${CODE_DIR:-${_CODE}}"

# --- Data, model, container ---
export DATADIR="${DATADIR:-}"
export MODELDIR="${MODELDIR:-}"
export MLPERF_MODELS_ROOT="${MLPERF_MODELS_ROOT:-}"
export LOGDIR="${LOGDIR:-${_CODE}/results}"
export CONT="${CONT:-}"
export CONFIG_FILE="${CONFIG_FILE:-config_MI350X_2x8x1.sh}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# --- RCCL / NCCL / UCX / MPI (Slurm + IB/RoCE) ---
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_GDR_LEVEL=0
export NCCL_IB_HCA=
export NCCL_IB_GID_INDEX=
export NCCL_SOCKET_IFNAME=
export GLOO_SOCKET_IFNAME=
export UCX_NET_DEVICES=
export HSA_NO_SCRATCH_RECLAIM=1
export OMPI_MCA_btl="^openib,ofi"
export OMPI_MCA_pml="ob1"

# --- Training config + run.sub defaults (skip when RUN_ENV_SKIP_CONFIG=1, e.g. Docker) ---
if [[ "${RUN_ENV_SKIP_CONFIG:-0}" != 1 ]]; then
  export NEXP="${NEXP:-10}"
  if [[ "${CONFIG_FILE}" = /* ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
  else
    # shellcheck source=/dev/null
    source "${_CODE}/${CONFIG_FILE}"
  fi
  # Must export (not only :=) so sbatch --export=ALL passes them into run.sub (set -u there).
  export WORK_DIR="${WORK_DIR:-/workspace/code}"
  export NCCL_TEST="${NCCL_TEST:-1}"
  export CLEAR_CACHES="${CLEAR_CACHES:-1}"
  export LOG_FREQ="${LOG_FREQ:-0}"
  export NCCL_TEST_WALLTIME="${NCCL_TEST_WALLTIME:-10}"
  export EXTRA_ASSETS="${EXTRA_ASSETS:-}"
  export DROPCACHE_CMD="${DROPCACHE_CMD:-sudo /sbin/sysctl vm.drop_caches=3}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  export WALLTIME_RUNANDTIME="${WALLTIME_RUNANDTIME:-${WALLTIME_MINUTES:-40}}"
  # Random per job unless caller sets SEED_BASE or SEED; run.sub still normalizes invalid/low bases.
  export SEED_BASE="${SEED_BASE:-${SEED:-$((RANDOM + RANDOM + 2))}}"
fi
