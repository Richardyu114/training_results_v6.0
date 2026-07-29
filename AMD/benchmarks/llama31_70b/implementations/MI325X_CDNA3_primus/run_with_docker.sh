#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euxo pipefail

# Change directory to the primus directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${MODELDIR:?MODELDIR not set}"
: "${LOGDIR:?LOGDIR not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
# No MLPerf benchmark definition exists for llama31_70b, so the compliance
# checker has no ruleset to validate against. Off by default here.
: "${CHECK_COMPLIANCE:=0}"
: "${MLPERF_RULESET:=6.0.0}"
: "${UTILITIES:="$(pwd)/../../utilities"}"

: "${CONT_NAME:=llama31_70b_pretraining}"
: "${NGPU:=1}"
: "${LOG_FREQ:=0}"
: "${HF_TOKEN:=""}"

# Other vars
readonly _config_file="${CONFIG_FILE:-config_${DGXSYSTEM}.sh}"
echo "CONFIG FILE: ${_config_file}"
if [[ ! -r "${_config_file}" ]]; then
    echo "ERROR: config file is not readable: ${_config_file}" >&2
    exit 2
fi
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${MODELDIR}:/model" "--volume=$(pwd):/workspace/code" "--volume=${LOGDIR}:/results")

if (( ${NNODES:-1} > 1 )); then
    source "${SCRIPT_DIR}/bnxt_rdma_overlay.sh"
    bnxt_rdma_prepare _cont_mounts "${DGXSYSTEM}" "${NNODES}" "${CONT}"
fi

# Setup directories
mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}/artifacts/"

# Get the config's exported variables. On multi-node runs, preserve caller
# network overrides while sourcing the config in the clean enumeration shell.
_config_source_env=(env -i CONFIG_FILE="${_config_file}")
_runtime_dist_env=()
if (( ${NNODES:-1} > 1 )); then
    mapfile -t _runtime_dist_env < <(
        compgen -e \
            | grep -E '^(NCCL_|TORCH_NCCL_|GLOO_|TORCH_DISTRIBUTED_DEBUG$)' \
            | sort -u \
            || true
    )
    for _dist_env_name in "${_runtime_dist_env[@]}"; do
        _config_source_env+=("${_dist_env_name}=${!_dist_env_name}")
    done
fi
if ! _config_export_names="$(
    "${_config_source_env[@]}" bash -c '. "${CONFIG_FILE}" && compgen -e'
)"; then
    echo "ERROR: failed to source config file: ${_config_file}" >&2
    exit 2
fi
mapfile -t _config_env < <(
    printf '%s\n' "${_config_export_names}" | grep -E -v '^(PWD|SHLVL)'
)
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

# Forward caller-only distributed overrides on multi-node runs.
if (( ${NNODES:-1} > 1 )); then
    _config_env+=("${_runtime_dist_env[@]}")
    mapfile -t _config_env < <(
        printf '%s\n' "${_config_env[@]}" | awk 'NF && !seen[$0]++'
    )
fi

echo "${_config_env[@]}"
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)
_base_config_env=("${_config_env[@]}")

# Cleanup container
cleanup_docker() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    else
        echo "Container ${_cont_name} does not exist. Skipping removal."
    fi
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
# Use DGXSYSTEM to determine hardware type (MI* = AMD/ROCm, otherwise NVIDIA)
if [[ "${DGXSYSTEM}" == MI* ]]; then
  echo "Using AMD/ROCm container flags"
  docker run --rm --init --detach \
      --net=host --uts=host --ipc=host \
      --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
      --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
      --security-opt=seccomp=unconfined \
      --group-add video \
      --privileged \
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


# Make sure container has time to finish initialization
sleep 5
docker exec "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    fi
    # Use existing SEED if set; otherwise use a new RANDOM value
    _run_config_env=("${_base_config_env[@]}" --env=SEED="${SEED:-$RANDOM}")
    echo "launching experiment using: ${_run_config_env[*]} ${_cont_name} /workspace/code/run_and_time.sh"
    docker exec "${_run_config_env[@]}" "${_cont_name}" bash -c '
      echo "[docker-env] BEGIN effective training environment"
      env | LC_ALL=C sort | while IFS= read -r entry; do
        printf "[docker-env] %s\n" "${entry}"
      done
      echo "[docker-env] END effective training environment"
      exec bash /workspace/code/run_and_time.sh
    '
  ) 2>&1 | grep --line-buffered -v "connected peer ranks" | tee "${_logfile_base}_${_experiment_index}.log"

  if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec "${_run_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
      || echo "WARNING: Compliance check failed for experiment ${_experiment_index} (non-blocking)"
  fi

done
