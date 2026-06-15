#!/bin/bash
# run_with_docker_4node.sh — multi-node MLPerf launcher for the Akash rack.
#
# Diverges from the 1-node `run_with_docker.sh` in three places:
#   1. env -i discovery forwards MASTER_ADDR/MASTER_PORT/NODE_RANK so configs
#      that mark them required (e.g. `${MASTER_ADDR:?...}` in 4-node configs)
#      don't abort under the clean env and silently drop the entire export list.
#   2. AMD/ROCm branch sets `--ulimit memlock=-1`, `--ulimit stack=67108864`,
#      and `--cap-add=IPC_LOCK` — required for `ibv_reg_mr` (RoCE GPU Direct
#      RDMA via ib_peer_mem). The 1-node path doesn't need this.
#   3. Mounts $CACHE_DIR at /npy_indices so the GPTDataset index cache lives
#      on shared storage (NFS) instead of <DATADIR>/cache (local-per-node).
#      Pair with the matching 4-node YAML which sets data_cache_path:/npy_indices.
#
# Required env (caller must set):
#   DGXSYSTEM, CONT, DATADIR, MODELDIR, LOGDIR, CONFIG_NAME, CACHE_DIR
#   MASTER_ADDR, NODE_RANK (per-node)

set -euxo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${MODELDIR:?MODELDIR not set}"
: "${LOGDIR:?LOGDIR not set}"
: "${CONFIG_NAME:?CONFIG_NAME not set}"
: "${CACHE_DIR:?CACHE_DIR not set (must be a shared/NFS path for multi-node)}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${NODE_RANK:?NODE_RANK not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=6.0.0}"
: "${UTILITIES:="$(pwd)/../../utilities"}"

: "${CONT_NAME:=mlperf_small_llm_pretraining_4node}"
: "${NGPU:=1}"
: "${LOG_FREQ:=0}"
: "${HF_TOKEN:=""}"
: "${MASTER_PORT:=29502}"

readonly _config_file="${CONFIG_NAME}"
echo "CONFIG_NAME: ${CONFIG_NAME}"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"

mkdir -p "${CACHE_DIR}"

_cont_mounts=(
  "--volume=${DATADIR}:/data"
  "--volume=${MODELDIR}:/model"
  "--volume=$(pwd):/workspace/code"
  "--volume=${LOGDIR}:/results"
  "--volume=${CACHE_DIR}:/npy_indices"
)

# Setup directories
mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}/artifacts/"

# Get list of envvars to pass to docker. Forward multi-node rendezvous vars
# into the discovery shell so configs that mark them required don't abort
# under `env -i` and silently drop the entire export list.
mapfile -t _config_env < <(env -i \
    MASTER_ADDR="${MASTER_ADDR:-}" \
    MASTER_PORT="${MASTER_PORT:-}" \
    NODE_RANK="${NODE_RANK:-}" \
    bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(MODELDIR)
_config_env+=(MODEL)
_config_env+=(DGXSYSTEM)
_config_env+=(PROFILER)
_config_env+=(LOGDIR)
_config_env+=(HIPBLASLT_LOG)
_config_env+=(GEMM_OFFLINE_TUNING)
_config_env+=(GEMM_USE_TUNING_RESULTS)
_config_env+=(HF_TOKEN)
_config_env+=(SEED)
_config_env+=(MLPERF_VERBOSE_LOGS)
_config_env+=(GLOO_SOCKET_IFNAME)
_config_env+=(NCCL_IB_DISABLE)
_config_env+=(NCCL_DEBUG)

echo "${_config_env[@]}"
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)
_base_config_env=("${_config_env[@]}")

# --- MLPerf power-submission sampler (per-trial sidecar, per-node) ---
# Each node runs its OWN run_with_docker_4node.sh under run.py / torchrun,
# so power_start spawns one local sidecar per node = N sidecars cluster-
# wide = N node_<idx>.txt files (one per node). The submission bundle's
# result_<i>/ is assembled post-trial by collecting each node's local
# node_<idx>.txt (rsync / fluentbit / whatever the deploy pipeline uses).
# Toggle via POWER_MONITOR (default 1; set 0 in caller env).
source "${SCRIPT_DIR}/../../../power/with_power.sh"

cleanup_docker() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    else
        echo "Container ${_cont_name} does not exist. Skipping removal."
    fi
}
cleanup_docker
# `power_stop` first so the sidecar gets SIGTERM (and writes its closing
# INTERVAL_END line) before the docker container is torn down. Both are
# idempotent / no-op-safe.
trap 'set -eux; power_stop; cleanup_docker' EXIT

# Setup container
if [[ "${DGXSYSTEM}" == MI* ]]; then
  echo "Using AMD/ROCm container flags (4-node: memlock + IPC_LOCK + RoCE bind mounts)"
  # RoCE/RDMA bind mounts:
  #   - /sys/class/infiniband: REQUIRED. Without it `ibv_devinfo` returns 0
  #     HCAs even though the image has ionic.driver + libionic-rdmav34.so;
  #     NCCL then silently falls back to TCP. Confirm with NCCL_DEBUG=INFO
  #     and grep for "NET/IB" vs "NET/Socket".
  #   - /etc/libibverbs.d and libionic-rdmav34.so: also bind-mounted from
  #     host so the container's RDMA provider always matches the host's.
  docker run --rm --init --detach \
      --net=host --uts=host --ipc=host \
      --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
      --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK \
      --security-opt=seccomp=unconfined \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --group-add video \
      --privileged \
      --volume=/sys/class/infiniband:/sys/class/infiniband:ro \
      --volume=/etc/libibverbs.d:/etc/libibverbs.d:ro \
      --volume=/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity
else
  echo "Using NVIDIA container flags"
  docker run --rm --init --detach \
      --net=host --uts=host \
      --ipc=host --gpus all \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --device=/dev/infiniband \
      --security-opt=seccomp=unconfined \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity
fi

sleep 5
docker exec "${_cont_name}" true

for _experiment_index in $(seq 1 "${NEXP}"); do
  # --- Start per-trial power sidecar (writes LOGDIR/power/result_<i>/node_<idx>.txt) ---
  power_start "${LOGDIR}/power/result_$((_experiment_index - 1))"

  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    fi
    if [[ -n "${SEED:-}" ]]; then
      _trial_seed=$((SEED + _experiment_index - 1))
    else
      _trial_seed=$RANDOM
    fi
    _run_config_env=("${_base_config_env[@]}" --env=SEED="${_trial_seed}")
    echo "launching experiment using: ${_run_config_env[*]} ${_cont_name} /workspace/code/run_and_time_4node.sh"
    docker exec "${_run_config_env[@]}" "${_cont_name}" bash /workspace/code/run_and_time_4node.sh
  ) | grep --line-buffered -v "connected peer ranks" | tee "${_logfile_base}_${_experiment_index}.log"

  power_stop

  if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec "${_run_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
      || echo "WARNING: Compliance check failed for experiment ${_experiment_index} (non-blocking)"
  fi

done
