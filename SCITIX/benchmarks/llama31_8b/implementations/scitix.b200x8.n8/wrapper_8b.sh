#!/bin/bash
set -euo pipefail

# ============================================================
# wrapper_8b.sh
# - Robust wrapper for NeMo/Megatron Llama3.1-8B MLPerf style runs on MPIJob
# - Fixes common K8s/MPIJob issues:
#   * OMPI_* -> RANK/LOCAL_RANK/WORLD_SIZE export early
#   * MASTER_ADDR/MASTER_PORT sane defaults (avoid 29500 collision)
#   * /preproc_data -> /data/8b mapping
#   * /npy_index -> /data/8b/npy_index mapping (shared + writable)
#   * /results -> shared results dir mapping with fallback
#   * tokenizer symlink to expected path
#   * creates per-rank patched run_and_time.sh to avoid collisions
# ============================================================

# ----------------------------
# 0) Config + defaults
# ----------------------------
CONFIG_FILE="${CONFIG_FILE:-/workspace/ft-llm-shared/config_B200_8x8x1xtp1pp1cp2_8b.sh}"
export SEED="${SEED:-42}"
echo "[Wrapper] SEED=${SEED}"
if [ -f "$CONFIG_FILE" ]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
else
  echo "[FATAL] Config not found: $CONFIG_FILE"
  exit 1
fi

# ----------------------------
# 1) Normalize distributed env (DO THIS EARLY)
# ----------------------------
# OpenMPI envs exist under MPIJob+mpirun
if [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then
  export WORLD_SIZE="${WORLD_SIZE:-${OMPI_COMM_WORLD_SIZE:-${SLURM_NTASKS:-1}}}"
  export RANK="${RANK:-${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}}"
  export LOCAL_RANK="${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}}"
  export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${OMPI_COMM_WORLD_LOCAL_SIZE:-${SLURM_NTASKS_PER_NODE:-1}}}"
else
  # Fallback: single-process
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export RANK="${RANK:-0}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-1}"
fi

# Group rank (node index)
export GROUP_RANK="${GROUP_RANK:-$(( RANK / LOCAL_WORLD_SIZE ))}"

# MASTER_ADDR: prefer existing; else guess "xxx-launcher" based on pod name pattern
if [ -z "${MASTER_ADDR:-}" ]; then
  export MASTER_ADDR="$(hostname | sed 's/-worker-.*//')-launcher"
  echo "[Wrapper] MASTER_ADDR not set, guessing: ${MASTER_ADDR}"
fi

# MASTER_PORT: avoid EADDRINUSE by picking a job-derived high port
if [ -z "${MASTER_PORT:-}" ]; then
  seed="${SLURM_JOB_ID:-${SLURM_JOBID:-${OMPI_JOBID:-$(date +%s)}}}"
  export MASTER_PORT=$(( 20000 + (seed % 20000) ))
fi

# Slurm job id fallback (MPIJob often doesn't set it)
export SLURM_JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-${OMPI_COMM_WORLD_JOBID:-${OMPI_JOBID:-no_slurm}}}}"

# Optional: better signal cleanup (prevents orphaned children on broken pipe)
trap 'echo "[Wrapper] Caught signal; killing child processes"; pkill -P $$ 2>/dev/null || true' INT TERM HUP

# ----------------------------
# 2) Paths (data/log/results)
# ----------------------------
export DATADIR="${DATADIR:-/data/8b}"
export LOGDIR="${LOGDIR:-/data/mlperf_training/logs}"
RESULTS_ROOT="${RESULTS_ROOT:-/data/mlperf_training/results}"

mkdir -p "$LOGDIR" || true
mkdir -p "$RESULTS_ROOT" || true

# /results mapping (many scripts expect /results)
if [ ! -e /results ]; then
  ln -s "$RESULTS_ROOT" /results 2>/dev/null || true
fi
# verify /results writable; if not, fall back
if ! (mkdir -p /results 2>/dev/null && touch /results/.rw_test 2>/dev/null); then
  echo "[Wrapper] /results not writable, falling back to /tmp/results"
  mkdir -p /tmp/results
  rm -rf /results 2>/dev/null || true
  ln -s /tmp/results /results 2>/dev/null || true
else
  rm -f /results/.rw_test 2>/dev/null || true
fi

# ----------------------------
# 3) Tokenizer link (expected by some NeMo configs)
# ----------------------------
EXPECTED_TOKENIZER_PATH="${EXPECTED_TOKENIZER_PATH:-/workspace/llm/nemo_tokenizer}"
REAL_TOKENIZER_PATH="${REAL_TOKENIZER_PATH:-${DATADIR}/tokenizer}"

if [ "${LOCAL_RANK:-0}" -eq 0 ]; then
  if [ -d "$REAL_TOKENIZER_PATH" ] && [ -f "$REAL_TOKENIZER_PATH/tokenizer.json" ]; then
    echo "[Wrapper] (Rank $RANK) Linking tokenizer: $EXPECTED_TOKENIZER_PATH -> $REAL_TOKENIZER_PATH"
    mkdir -p "$(dirname "$EXPECTED_TOKENIZER_PATH")"
    rm -rf "$EXPECTED_TOKENIZER_PATH"
    ln -s "$REAL_TOKENIZER_PATH" "$EXPECTED_TOKENIZER_PATH"
  else
    echo "[FATAL] Tokenizer not found at: $REAL_TOKENIZER_PATH (expect tokenizer.json)"
    ls -lah "$REAL_TOKENIZER_PATH" 2>/dev/null || true
    exit 1
  fi
else
  # wait briefly for the symlink
  for _ in $(seq 1 60); do
    [ -e "$EXPECTED_TOKENIZER_PATH" ] && break
    sleep 1
  done
fi

# ----------------------------
# 4) Dataset path mapping: /preproc_data -> ${DATADIR}
# ----------------------------
if [ "${LOCAL_RANK:-0}" -eq 0 ]; then
  # confirm dataset exists
  if [ -f "${DATADIR}/c4-train.en_6_text_document.bin" ]; then
    TARGET="${DATADIR}"
  elif [ -f "${DATADIR}/preproc_data/c4-train.en_6_text_document.bin" ]; then
    TARGET="${DATADIR}/preproc_data"
  else
    echo "[FATAL] Cannot find c4-train.en_6_text_document.bin under ${DATADIR}"
    echo "Try:"
    echo "  ls -lah ${DATADIR} | head"
    echo "  ls -lah ${DATADIR}/preproc_data | head"
    exit 1
  fi

  if [ -e /preproc_data ] && [ ! -L /preproc_data ]; then
    echo "[Wrapper] /preproc_data exists and is not a symlink; not touching it."
  elif [ ! -e /preproc_data ]; then
    ln -s "${TARGET}" /preproc_data
    echo "[Wrapper] Linked /preproc_data -> ${TARGET}"
  else
    echo "[Wrapper] /preproc_data already exists (likely symlink)."
  fi
fi

# ----------------------------
# 5) Megatron index cache: /npy_index -> ${DATADIR}/npy_index
# ----------------------------
NPY_SHARED="${DATADIR}/npy_index"
mkdir -p "${NPY_SHARED}" || { echo "[FATAL] cannot mkdir ${NPY_SHARED}"; exit 1; }
chmod 777 "${NPY_SHARED}" 2>/dev/null || true

if [ "${LOCAL_RANK:-0}" -eq 0 ]; then
  # replace non-symlink /npy_index if present
  if [ -e /npy_index ] && [ ! -L /npy_index ]; then
    echo "[Wrapper] /npy_index exists and is not a symlink; replacing it"
    rm -rf /npy_index || { echo "[FATAL] cannot remove /npy_index"; exit 1; }
  fi
  if [ ! -e /npy_index ]; then
    ln -s "${NPY_SHARED}" /npy_index || { echo "[FATAL] cannot create /npy_index symlink"; exit 1; }
  fi
fi

# wait for /npy_index to appear on non-local-rank0
for _ in $(seq 1 60); do
  [ -e /npy_index ] && break
  sleep 1
done

# strict check on rank0
if [ "${RANK:-0}" -eq 0 ]; then
  echo "[Wrapper] /npy_index -> $(readlink -f /npy_index || echo /npy_index)"
  touch /npy_index/.rw_test || { echo "[FATAL] /npy_index not writable"; exit 1; }
  rm -f /npy_index/.rw_test || true
fi

# ----------------------------
# 6) NCCL network (optional, keep minimal; do NOT crash if wrong)
# ----------------------------
# If you know your HCAs, set them; otherwise comment out.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
#export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_IB_HCA='=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7'
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,NET

# Modern torch deprecation cleanup (optional)
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
unset NCCL_ASYNC_ERROR_HANDLING || true
unset TORCH_NCCL_AVOID_RECORD_STREAMS || true

# ----------------------------
# 7) Create per-rank run_and_time script and patch known fragile bits
# ----------------------------
ORIGIN_SCRIPT="${ORIGIN_SCRIPT:-./run_and_time.sh}"
if [ ! -f "$ORIGIN_SCRIPT" ]; then
  echo "[FATAL] Cannot find origin script: $ORIGIN_SCRIPT (cwd=$(pwd))"
  exit 1
fi

TARGET_SCRIPT="./run_and_time_patched_${RANK}.sh"
echo "[Wrapper] Creating unique script for Rank ${RANK}: $TARGET_SCRIPT"
cp "$ORIGIN_SCRIPT" "$TARGET_SCRIPT"
chmod +x "$TARGET_SCRIPT"

# Patch: env dump should not kill the job if /results or job id are missing
# (your original script had: env > /results/container-env-"$SLURM_JOB_ID".log)
sed -i 's|env > /results/container-env-"$SLURM_JOB_ID".log|env > /results/container-env-"${SLURM_JOB_ID:-no_slurm}".log || true|g' "$TARGET_SCRIPT" || true

# ----------------------------
# X) Cache clear (MLPerf-style) - optional
# ----------------------------
CLEAR_CACHES="${CLEAR_CACHES:-1}"
DROPCACHE_CMD="${DROPCACHE_CMD:-sudo /sbin/sysctl vm.drop_caches=3}"

cache_clear_all_nodes() {
  # 只让每个节点的 local_rank0 执行一次
  if [ "${LOCAL_RANK:-0}" -ne 0 ]; then
    return 0
  fi

  host="$(hostname)"
  echo "[CacheClear] ${host} sync_start"
  sync && echo "[CacheClear] ${host} sync_done"

  cache_before="$(awk '/^Cached:/ {print $2}' /proc/meminfo || echo 0)"
  echo "[CacheClear] ${host} cached_before=${cache_before}kB"

  # 尝试 drop caches（无权限不让任务失败）
  if ${DROPCACHE_CMD} 2>/dev/null; then
    echo "[CacheClear] ${host} drop_caches ok"
  else
    echo "[CacheClear] ${host} drop_caches failed (no sudo?). Continuing."
  fi

  cache_after="$(awk '/^Cached:/ {print $2}' /proc/meminfo || echo 0)"
  echo "[CacheClear] ${host} cached_after=${cache_after}kB"

  # 记录 MLPerf mllog 事件（同样不让失败影响训练）
  python - <<'PY' 2>/dev/null || true
from mlperf_common.callbacks import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)
PY
}

if [ "${CLEAR_CACHES}" -eq 1 ]; then
  echo "[Wrapper] CLEAR_CACHES=1 -> performing cache clear (per-node, local_rank0)"
  cache_clear_all_nodes
fi

# 为了保证所有 rank 在清 cache 之后再进入训练，做一次简易 barrier
# （不依赖 MPI API：用等待文件在共享目录实现）
BARRIER_DIR="${BARRIER_DIR:-/results/barriers}"
mkdir -p "${BARRIER_DIR}" || true
barrier_tag="${SLURM_JOB_ID:-${OMPI_JOBID:-job}}_cacheclear"
touch "${BARRIER_DIR}/${barrier_tag}.rank${RANK}" 2>/dev/null || true
# 等待所有 rank 都到达（WORLD_SIZE 个文件）
if [ "${WORLD_SIZE:-1}" -gt 1 ]; then
  for _ in $(seq 1 300); do
    n="$(ls -1 "${BARRIER_DIR}/${barrier_tag}.rank"* 2>/dev/null | wc -l || true)"
    [ "$n" -ge "${WORLD_SIZE}" ] && break
    sleep 1
  done
fi
# ----------------------------
# 8) Launch
# ----------------------------
echo "----------------------------------------------------------------"
echo "[Wrapper] Launching Rank $RANK on $(hostname)"
echo "[Wrapper] WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE"
echo "[Wrapper] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[Wrapper] DATADIR=$DATADIR LOGDIR=$LOGDIR RESULTS_ROOT=$RESULTS_ROOT"
echo "----------------------------------------------------------------"

exec bash "$TARGET_SCRIPT"

