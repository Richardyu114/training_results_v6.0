#!/bin/bash
# Two-node launcher for llama31_8b. Run once on node0.
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

: "${NODE0_IP:?set NODE0_IP (node0 IP, reachable from node1)}"
: "${NODE1_IP:?set NODE1_IP (node1 IP, reachable over SSH)}"
: "${CONT:?set CONT (same image tag on both nodes)}"
: "${DATADIR:?set DATADIR}"
: "${MODELDIR:?set MODELDIR}"
: "${LOGDIR:?set LOGDIR}"
: "${SEED:?set SEED (same value on both nodes)}"

SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
MASTER_PORT="${MASTER_PORT:-29502}"
MASTER_ADDR="$NODE0_IP"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
DGXSYSTEM_2N="${DGXSYSTEM_2N:-MI308X_2x8x1}"
LOG_PREFIX="${LOG_PREFIX:-run_2node}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
CONT_NAME_BASE="${CONT_NAME_BASE:-${CONT_NAME:-mlperf_llama31_8b}}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$ ]] || {
  echo "[2node] ERROR: invalid RUN_ID: ${RUN_ID}" >&2
  exit 1
}
[[ "${CONT_NAME_BASE}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "[2node] ERROR: invalid CONT_NAME_BASE: ${CONT_NAME_BASE}" >&2
  exit 1
}

NODE0_CONT_NAME="${CONT_NAME_BASE}_${RUN_ID}_n0"
NODE1_CONT_NAME="${CONT_NAME_BASE}_${RUN_ID}_n1"
NODE0_LOG="${LOG_PREFIX}_${RUN_ID}_node0.log"
NODE1_LOG="${LOG_PREFIX}_${RUN_ID}_node1.log"

SSH=(ssh -p "$SSH_PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=10 -o ConnectionAttempts=1)

PRIMUS_TRAIN_ITERS="${PRIMUS_TRAIN_ITERS:-50}"
MLPERF_VERBOSE_LOGS="${MLPERF_VERBOSE_LOGS:-1}"
NEXP="${NEXP:-1}"
CLEAR_CACHES="${CLEAR_CACHES:-1}"
EXP="${EXP:-}"
WARMUP_RECIPE="${WARMUP_RECIPE:-}"
PRIMUS_LR="${PRIMUS_LR:-}"
PRIMUS_MIN_LR="${PRIMUS_MIN_LR:-}"

[[ "$CLEAR_CACHES" == 0 || "$CLEAR_CACHES" == 1 ]] || {
  echo "[2node] ERROR: CLEAR_CACHES must be 0 or 1" >&2
  exit 1
}

# Preserve caller overrides on both nodes. The two optional RDMA paths normally
# stay unset so each Docker daemon host discovers its own matching pair.
mapfile -t CALLER_DIST_ENV < <(
  compgen -e \
    | grep -E '^(NCCL_|TORCH_NCCL_|GLOO_|TORCH_DISTRIBUTED_DEBUG$|BNXT_RDMA_LIBIBVERBS_HOST_PATH$|BNXT_RDMA_PROVIDER_HOST_PATH$|MLPERF_HOST_LIBIBVERBS_PATH$|MLPERF_HOST_BNXT_PROVIDER_PATH$)' \
    | sort -u || true
)

shell_assignment() {
  printf '%s=%q' "$1" "$2"
}

build_caller_dist_exports() {
  local name output=""
  for name in "${CALLER_DIST_ENV[@]}"; do
    output+=" $(shell_assignment "$name" "${!name}")"
  done
  printf '%s' "$output"
}

CALLER_DIST_EXPORTS="$(build_caller_dist_exports)"
if ! docker info >/dev/null 2>&1; then
  echo "[2node] ERROR: the node0 user cannot access the Docker daemon" >&2
  exit 1
fi
DOCKER_LOCAL=(docker)

echo "==================================================================="
echo "[2node] node0=$NODE0_IP  node1=$NODE1_IP  master=$MASTER_ADDR:$MASTER_PORT"
echo "[2node] image=$CONT  seed=$SEED  iters=$PRIMUS_TRAIN_ITERS"
echo "[2node] containers: $NODE0_CONT_NAME / $NODE1_CONT_NAME"
echo "[2node] logs: $NODE0_LOG / $NODE1_LOG"
echo "==================================================================="

if ! "${DOCKER_LOCAL[@]}" images --format '{{.Repository}}:{{.Tag}}' | grep -Fqx -- "$CONT"; then
  echo "[2node] ERROR: image $CONT not found on node0" >&2
  exit 1
fi

printf -v REMOTE_IMAGE_CHECK \
  'docker info >/dev/null 2>&1 && docker images --format={{.Repository}}:{{.Tag}} | grep -Fqx -- %q' \
  "$CONT"
if ! "${SSH[@]}" "$SSH_USER@$NODE1_IP" "$REMOTE_IMAGE_CHECK"; then
  echo "[2node] ERROR: image $CONT not found on node1, or SSH/docker failed" >&2
  exit 1
fi

printf -v REMOTE_REPO_CHECK 'test -d %q' "$REPO_DIR"
if ! "${SSH[@]}" "$SSH_USER@$NODE1_IP" "$REMOTE_REPO_CHECK"; then
  echo "[2node] ERROR: repo $REPO_DIR not found on node1" >&2
  exit 1
fi

build_cmd() {  # $1=node rank, $2=container name
  local rank="$1" container_name="$2" cmd
  printf -v cmd 'cd %q' "$REPO_DIR"
  cmd+=" && export $(shell_assignment DATADIR "$DATADIR")"
  cmd+=" $(shell_assignment MODELDIR "$MODELDIR")"
  cmd+=" $(shell_assignment LOGDIR "$LOGDIR")"
  cmd+=" $(shell_assignment CONT "$CONT")"
  cmd+=" $(shell_assignment SEED "$SEED")"
  cmd+=" $(shell_assignment NEXP "$NEXP")"
  cmd+=" $(shell_assignment CLEAR_CACHES "$CLEAR_CACHES")"
  cmd+=" $(shell_assignment NODE_RANK "$rank")"
  cmd+=" $(shell_assignment MASTER_ADDR "$MASTER_ADDR")"
  cmd+=" $(shell_assignment CONT_NAME "$container_name")"
  cmd+="$CALLER_DIST_EXPORTS"
  cmd+=" && source $(printf '%q' "config_${DGXSYSTEM_2N}.sh")"
  cmd+=" && export $(shell_assignment MASTER_PORT "$MASTER_PORT")"
  cmd+=" $(shell_assignment NODE_RANK "$rank")"
  cmd+=" $(shell_assignment CONT_NAME "$container_name")"
  cmd+=" $(shell_assignment PRIMUS_TRAIN_ITERS "$PRIMUS_TRAIN_ITERS")"
  cmd+=" $(shell_assignment MLPERF_VERBOSE_LOGS "$MLPERF_VERBOSE_LOGS")"
  cmd+="$CALLER_DIST_EXPORTS"
  [[ -n "$EXP" ]] && cmd+=" $(shell_assignment EXP "$EXP")"
  [[ -n "$WARMUP_RECIPE" ]] && cmd+=" $(shell_assignment WARMUP_RECIPE "$WARMUP_RECIPE")"
  [[ -n "$PRIMUS_LR" ]] && cmd+=" $(shell_assignment PRIMUS_LR "$PRIMUS_LR")"
  [[ -n "$PRIMUS_MIN_LR" ]] && cmd+=" $(shell_assignment PRIMUS_MIN_LR "$PRIMUS_MIN_LR")"
  cmd+=" && bash run_with_docker.sh"
  printf '%s' "$cmd"
}

cleanup_both_containers() {
  set +e
  echo "[2node] stopping/removing both training containers"
  "${DOCKER_LOCAL[@]}" container stop "$NODE0_CONT_NAME" >/dev/null 2>&1
  "${DOCKER_LOCAL[@]}" container rm -f "$NODE0_CONT_NAME" >/dev/null 2>&1

  local remote_cleanup
  printf -v remote_cleanup \
    'docker container stop %q >/dev/null 2>&1 || true; docker container rm -f %q >/dev/null 2>&1 || true' \
    "$NODE1_CONT_NAME" "$NODE1_CONT_NAME"
  "${SSH[@]}" "$SSH_USER@$NODE1_IP" "$remote_cleanup" >/dev/null 2>&1 || true
  set -e
}

NODE0_PID=""
NODE1_PID=""
stop_launchers() {
  [[ -z "$NODE0_PID" ]] || kill "$NODE0_PID" 2>/dev/null || true
  [[ -z "$NODE1_PID" ]] || kill "$NODE1_PID" 2>/dev/null || true
  [[ -z "$NODE0_PID" ]] || wait "$NODE0_PID" 2>/dev/null || true
  [[ -z "$NODE1_PID" ]] || wait "$NODE1_PID" 2>/dev/null || true
}

handle_signal() {
  local rc="$1"
  trap - HUP INT TERM
  cleanup_both_containers
  stop_launchers
  exit "$rc"
}
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

NODE0_CMD="$(build_cmd 0 "$NODE0_CONT_NAME")"
NODE1_CMD="$(build_cmd 1 "$NODE1_CONT_NAME")"

echo "[2node] starting node0..."
(
  set +e
  set -o pipefail
  bash -c "$NODE0_CMD" 2>&1 | tee "$NODE0_LOG"
  status=("${PIPESTATUS[@]}")
  (( status[0] != 0 )) && exit "${status[0]}"
  exit "${status[1]}"
) &
NODE0_PID=$!

sleep 3

echo "[2node] starting node1..."
"${SSH[@]}" "$SSH_USER@$NODE1_IP" "$NODE1_CMD" >"$NODE1_LOG" 2>&1 &
NODE1_PID=$!

set +e
FINISHED_PID=""
wait -n -p FINISHED_PID "$NODE0_PID" "$NODE1_PID"
FIRST_RC=$?
set -e

if (( FIRST_RC != 0 )); then
  echo "[2node] launcher pid=$FINISHED_PID failed rc=$FIRST_RC; stopping both nodes" >&2
  cleanup_both_containers
  stop_launchers
  exit "$FIRST_RC"
fi

if [[ "$FINISHED_PID" == "$NODE0_PID" ]]; then
  REMAINING_PID="$NODE1_PID"
else
  REMAINING_PID="$NODE0_PID"
fi

set +e
wait "$REMAINING_PID"
SECOND_RC=$?
set -e
if (( SECOND_RC != 0 )); then
  echo "[2node] peer launcher failed rc=$SECOND_RC; stopping both nodes" >&2
  cleanup_both_containers
  stop_launchers
  exit "$SECOND_RC"
fi

trap - HUP INT TERM
echo "[2node] DONE: node0 and node1 both exited successfully"
echo "[2node] logs: $NODE0_LOG / $NODE1_LOG"
