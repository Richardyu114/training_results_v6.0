#!/bin/bash
# Bare-metal single-node launcher for Llama-3.1-8B on 8x H200 (no Slurm/Pyxis).
#
# The official NVIDIA submission drives run_and_time.sh through Slurm+Pyxis (run.sub). This script
# runs the same containerized training on a single node with plain Docker instead: it starts the
# training image, forwards the selected config's environment into the container, and invokes
# run_and_time.sh once with LOCAL_WORLD_SIZE=1 so that run_and_time.sh takes its interactive
# single-node branch and fans out 8 ranks itself via `torchrun --nproc_per_node=8`.
#
# Usage:
#   CONT=mlperf-nvidia:llama31_8b-pyt \
#   DATADIR=/path/to/data \      # must contain 8b/ (see data_scripts/download_8b.sh)
#   LOGDIR=/path/to/logs \
#   CONFIG_FILE=config_H200_1x8x2xtp1pp1cp1_8b_fp8.sh \   # or ..._bf16.sh
#   bash run_with_docker.sh
#
# Quick smoke test (50 steps, does not reach target): prepend MAX_STEPS=50.

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# Vars without defaults
: "${CONT:?CONT not set (training image, e.g. mlperf-nvidia:llama31_8b-pyt)}"
: "${DATADIR:?DATADIR not set (host dir containing 8b/)}"
: "${LOGDIR:?LOGDIR not set (host dir for logs/results)}"
: "${CONFIG_FILE:?CONFIG_FILE not set (e.g. config_H200_1x8x2xtp1pp1cp1_8b_fp8.sh)}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CONT_NAME:=mlperf_llama31_8b}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

if [[ ! -r "${CONFIG_FILE}" ]]; then
    echo "ERROR: config file is not readable: ${CONFIG_FILE}" >&2
    exit 2
fi
echo "CONFIG FILE: ${CONFIG_FILE}"

# --- Resolve mounts (mirrors config_mounts.sh, single-node real-data case) ---
# download_8b.sh + cleanup_8b.sh lay the data out as ${DATADIR}/8b/{*.bin,*.idx,tokenizer/}.
_tokenizer="${DATADIR}/8b/tokenizer"
_preproc="${DATADIR}/8b"
_npy_index_dir="${LOGDIR}/${DATESTAMP}_npy_index"
_mem_dump_dir="${LOGDIR}/mem_dump"
( umask 0002; mkdir -p "${LOGDIR}" "${_npy_index_dir}" "${_mem_dump_dir}" )
# Because /workspace/llm is bind-mounted from ${SCRIPT_DIR}, the nested tokenizer mount lands at
# ${SCRIPT_DIR}/nemo_tokenizer; ensure that mountpoint dir exists on the host (git-ignored).
mkdir -p "${SCRIPT_DIR}/nemo_tokenizer"

_cont_mounts=(
    # Mount this directory over the image's /workspace/llm so edits to the training code
    # (pretrain.py, configs, run_and_time.sh, ...) take effect without rebuilding the image.
    # The compiled embedding_lib CUDA extension lives in site-packages, not here, so it is
    # unaffected; the Python wrapper is imported from embedding_lib/ in this dir, which is present.
    "--volume=${SCRIPT_DIR}:/workspace/llm"
    "--volume=${LOGDIR}:/results"
    "--volume=${_npy_index_dir}:/npy_index"
    "--volume=${_mem_dump_dir}:/mem_dump"
    "--volume=${_tokenizer}:/workspace/llm/nemo_tokenizer:ro"
    "--volume=${_preproc}:/preproc_data:ro"
)

# --- Collect the config's exported variable names, then forward them into the container ---
# Source the config in a clean subshell (like the official launcher) and enumerate its exports, so
# every MINIBS/TP/FP8/NCCL/... it sets is forwarded by name and the container reads the real value
# from this shell's environment via `docker exec --env=NAME`.
if ! _config_export_names="$(
    env -i CONFIG_FILE="${CONFIG_FILE}" PATH="${PATH}" \
        bash -c '. "${CONFIG_FILE}" && compgen -e'
)"; then
    echo "ERROR: failed to source config file: ${CONFIG_FILE}" >&2
    exit 2
fi

# Also load the config into THIS shell so the forwarded names have values to read.
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

mapfile -t _config_env < <(
    printf '%s\n' "${_config_export_names}" | grep -E -v '^(PWD|SHLVL|_|CONFIG_FILE)$'
)
# Extras the container needs that are not part of the config file itself.
# MASTER_ADDR/MASTER_PORT are passed explicitly below in _bootstrap_env.
_config_env+=(DATADIR MODEL_NAME MODEL_FRAMEWORK DGXSYSTEM LOGDIR)

# torchrun bootstrap vars: run_and_time.sh requires these to be set, and with LOCAL_WORLD_SIZE=1 it
# takes the single-node branch and runs `torchrun --nproc_per_node=${DGXNGPU}` itself.
_bootstrap_env=(
    "--env=RANK=0"
    "--env=LOCAL_RANK=0"
    "--env=WORLD_SIZE=1"
    "--env=LOCAL_WORLD_SIZE=1"
    "--env=MASTER_ADDR=${MASTER_ADDR}"
    "--env=MASTER_PORT=${MASTER_PORT}"
)

export MODEL_NAME="${MODEL_NAME:-llama31_8b}"
export MODEL_FRAMEWORK="${MODEL_FRAMEWORK:-pytorch}"

mapfile -t _env_flags < <(for v in "${_config_env[@]}"; do echo "--env=${v}"; done)

# --- Container lifecycle ---
cleanup_docker() {
    docker container rm -f "${CONT_NAME}" >/dev/null 2>&1 || true
}
cleanup_docker
trap cleanup_docker EXIT

docker run --rm --init --detach \
    --net=host --uts=host --ipc=host \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=64g \
    --security-opt=seccomp=unconfined \
    --name="${CONT_NAME}" "${_cont_mounts[@]}" \
    -e IMAGE_NAME="${CONT}" \
    "${CONT}" sleep infinity

sleep 5
docker exec "${CONT_NAME}" true

# --- Run experiment(s) ---
_logfile_base="${LOGDIR}/${DATESTAMP}"
for _exp in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_exp} of ${NEXP}"
        if [[ "${CLEAR_CACHES}" == 1 ]]; then
            sync && (sudo /sbin/sysctl vm.drop_caches=3 || echo "WARN: could not drop caches (need sudo); continuing")
        fi
        _seed_env=("--env=SEED=${SEED:-$RANDOM}")
        docker exec \
            "${_env_flags[@]}" "${_bootstrap_env[@]}" "${_seed_env[@]}" \
            --workdir=/workspace/llm \
            "${CONT_NAME}" bash /workspace/llm/run_and_time.sh
    ) 2>&1 | tee "${_logfile_base}_${_exp}.log"
done
