#!/bin/bash
set -euxo pipefail

cd "${REPO_DIR}"

: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${LOGDIR:?LOGDIR not set}"

: "${NEXP:=1}"
: "${CLEAR_CACHES:=1}"
: "${CONT_NAME:=mlperf_flux1}"
: "${LOG_FREQ:=0}"
: "${ROOT_SEED:=${ROOT_SEED}}"
: "${NGPU:=1}"
: "${STAGE_TO_LOCAL_NVME:=0}"
: "${HYBRID_LOCAL_OR_NFS:=0}"
: "${STAGE_FANOUT:=0}"
: "${LOCAL_DATADIR_BASE:=/mnt/localdisk}"
: "${LOCAL_STAGE_WORKERS:=64}"
: "${LOCAL_STAGE_MIN_BYTES:=2000000000000}"
: "${ULIMIT_NOFILE:=1048576}"
: "${RUNTIME_TMP_DIR:=/mnt/localdisk/flux1_runtime_tmp}"

: "${PROFILER:=""}"
: "${TORCHPROF_OUTPUT_DIR:=""}"
: "${PROF_WARMUP_STEPS:=5}"
: "${PROF_ACTIVE_STEPS:=2}"
: "${ROCPROF_PROFILER:='none'}"
: "${ROCPROF:='/opt/rocm/bin/rocprofv3'}"
: "${HIPBLASLT_LOG:=0}"
: "${GEMM_OFFLINE_TUNING:=0}"
: "${GEMM_USE_TUNING_RESULTS:=0}"
: "${RUN_OPTUNA:=0}"
: "${OPTUNA_CONFIG:=''}"
: "${ENABLE_MEMORY_PROFILING:=0}"
: "${POWER_PROFILING:=0}"
: "${ENABLE_TRAINING_METRICS_COLLECTION:=0}"

echo "ROOT SEED IS: $ROOT_SEED"
echo "Host open files limit on $(hostname): $(ulimit -n)"

readonly _config_file="${CONFIG_DIR}/config_${DGXSYSTEM}.sh"
readonly _cont_name="${CONT_NAME}"

stage_dir_has_data() {
    local dir="$1"
    local bytes
    local entries
    [[ -d "${dir}" ]] || return 1
    bytes="$(du -sb "${dir}" 2>/dev/null | awk '{print $1}')"
    entries="$(find "${dir}" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"
    [[ "${bytes:-0}" =~ ^[0-9]+$ ]] || return 1
    [[ "${entries:-0}" =~ ^[0-9]+$ ]] || return 1
    (( bytes >= LOCAL_STAGE_MIN_BYTES && entries > 0 ))
}

stage_is_ready_local() {
    local dir="$1"
    local marker="$2"
    [[ -f "${marker}" ]] || stage_dir_has_data "${dir}"
}

stage_is_ready_remote() {
    local host="$1"
    local dir="$2"
    local marker="$3"
    local ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
    ssh "${ssh_opts[@]}" "${host}" "bash -lc '[[ -f \"${marker}\" ]] || { [[ -d \"${dir}\" ]] && bytes=\$(du -sb \"${dir}\" 2>/dev/null | awk \"{print \\\$1}\") && entries=\$(find \"${dir}\" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l) && [[ \"\${bytes:-0}\" =~ ^[0-9]+$ ]] && [[ \"\${entries:-0}\" =~ ^[0-9]+$ ]] && (( bytes >= ${LOCAL_STAGE_MIN_BYTES} && entries > 0 )); }'" >/dev/null 2>&1
}

wait_for_stage_status() {
    local done_file="$1"
    local fail_file="$2"
    while true; do
        if [[ -f "${done_file}" ]]; then
            return 0
        fi
        if [[ -f "${fail_file}" ]]; then
            cat "${fail_file}" >&2 || true
            return 1
        fi
        sleep 5
    done
}

wait_for_direct_stage() {
    local done_file="$1"
    local fail_file="$2"
    local stage_state_dir="$3"
    shift 3
    local -a nodes=("$@")
    local node

    while true; do
        if [[ -f "${fail_file}" ]]; then
            cat "${fail_file}" >&2 || true
            return 1
        fi

        local all_done=1
        for node in "${nodes[@]}"; do
            if [[ ! -f "${stage_state_dir}/${node}.done" ]]; then
                all_done=0
                break
            fi
        done

        if (( all_done == 1 )); then
            touch "${done_file}"
            return 0
        fi

        sleep 5
    done
}

seed_sync_local() {
    local src_dir="$1"
    local stage_dir="$2"
    local stage_workers="$3"
    local tmp_stage_dir="${stage_dir}.seed.${SLURM_JOB_ID}.tmp"

    echo "=== Seed staging dataset from shared storage on $(hostname) ==="
    echo "Source: ${src_dir}"
    echo "Destination: ${stage_dir}"
    echo "Workers: ${stage_workers}"

    sudo rm -rf "${tmp_stage_dir}"
    sudo mkdir -p "${tmp_stage_dir}"
    sudo chmod 777 "${tmp_stage_dir}"

    if command -v rsync >/dev/null 2>&1; then
        if (( stage_workers > 1 )); then
            local -a src_dirs=()
            local -a src_files=()
            mapfile -d '' -t src_dirs < <(find "${src_dir}" -type d -printf '%P\0')
            mapfile -d '' -t src_files < <(find "${src_dir}" \( -type f -o -type l \) -printf '%P\0')
            local rel_path
            for rel_path in "${src_dirs[@]}"; do
                [[ -z "${rel_path}" ]] && continue
                sudo mkdir -p "${tmp_stage_dir}/${rel_path}"
            done
            printf '%s\0' "${src_files[@]}" | \
                xargs -0 -r -P "${stage_workers}" -I{} \
                bash -c 'set -euo pipefail; src_root="$1"; dst_root="$2"; rel_path="$3"; sudo rsync -a --numeric-ids "${src_root}/${rel_path}" "${dst_root}/${rel_path}"' \
                _ "${src_dir}" "${tmp_stage_dir}" {}
        else
            sudo rsync -a --delete-delay --numeric-ids --info=stats2 "${src_dir}/" "${tmp_stage_dir}/"
        fi
    else
        sudo cp -a "${src_dir}/." "${tmp_stage_dir}/"
    fi

    sudo rm -rf "${stage_dir}"
    sudo mv "${tmp_stage_dir}" "${stage_dir}"
    sudo chmod -R a+rX "${stage_dir}"
}

fanout_copy_stage() {
    local src_host="$1"
    local dst_host="$2"
    local stage_dir="$3"
    local marker_rel="$4"
    local ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
    local tmp_stage_dir="${stage_dir}.fanout.${SLURM_JOB_ID}.tmp"

    echo "Fan-out copy ${src_host} -> ${dst_host}"
    if [[ "${src_host}" == "$(hostname)" ]]; then
        sudo tar -C "${stage_dir}" -cf - . | \
            ssh "${ssh_opts[@]}" "${dst_host}" \
            "sudo rm -rf '${tmp_stage_dir}' '${stage_dir}' && sudo mkdir -p '${tmp_stage_dir}' && sudo tar -C '${tmp_stage_dir}' -xf - && sudo mv '${tmp_stage_dir}' '${stage_dir}' && sudo chmod -R a+rX '${stage_dir}' && sudo touch '${stage_dir}/${marker_rel}'"
    else
        ssh "${ssh_opts[@]}" "${src_host}" "sudo tar -C '${stage_dir}' -cf - ." | \
            ssh "${ssh_opts[@]}" "${dst_host}" \
            "sudo rm -rf '${tmp_stage_dir}' '${stage_dir}' && sudo mkdir -p '${tmp_stage_dir}' && sudo tar -C '${tmp_stage_dir}' -xf - && sudo mv '${tmp_stage_dir}' '${stage_dir}' && sudo chmod -R a+rX '${stage_dir}' && sudo touch '${stage_dir}/${marker_rel}'"
    fi
}

stage_dataset_to_local_nvme() {
    if [[ "${STAGE_TO_LOCAL_NVME}" != "1" ]]; then
        return 0
    fi

    local src_dir="${DATADIR%/}"
    local local_base="${LOCAL_DATADIR_BASE%/}"
    local local_user="${USER:-${SLURM_JOB_USER:-flux1}}"
    local default_stage_dir="${local_base}/${local_user}/$(basename "${src_dir}")"
    local stage_dir="${LOCAL_DATADIR:-${default_stage_dir}}"
    local stage_workers="${LOCAL_STAGE_WORKERS}"
    local marker_rel=".flux_stage_ready_v1"
    local marker_path="${stage_dir}/${marker_rel}"
    local stage_state_prefix="local_stage"
    [[ "${STAGE_FANOUT}" == "1" ]] && stage_state_prefix="fanout_stage"
    local stage_state_dir="${LOGDIR%/}/${stage_state_prefix}_${SLURM_JOB_ID}"
    local done_file="${stage_state_dir}/done"
    local fail_file="${stage_state_dir}/failed"
    local seed_file="${stage_state_dir}/seed"
    local -a nodes=()

    if [[ ! -d "${src_dir}" ]]; then
        echo "Source DATADIR does not exist: ${src_dir}" >&2
        exit 1
    fi
    if [[ ! -d "${local_base}" ]]; then
        echo "Local staging base does not exist on $(hostname): ${local_base}" >&2
        exit 1
    fi
    if ! [[ "${stage_workers}" =~ ^[1-9][0-9]*$ ]]; then
        echo "LOCAL_STAGE_WORKERS must be a positive integer, got: ${stage_workers}" >&2
        exit 1
    fi

    mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    mkdir -p "${stage_state_dir}"

    if [[ "${STAGE_FANOUT}" != "1" ]]; then
        local node_name
        local node_done
        node_name="$(hostname)"
        node_done="${stage_state_dir}/${node_name}.done"

        echo "=== Direct NFS staging mode on ${node_name} ==="
        echo "Fan-out disabled: STAGE_FANOUT=${STAGE_FANOUT}"
        if stage_is_ready_local "${stage_dir}" "${marker_path}"; then
            echo "Local dataset already staged on ${node_name}: ${stage_dir}"
            sudo touch "${marker_path}"
        else
            seed_sync_local "${src_dir}" "${stage_dir}" "${stage_workers}"
            sudo touch "${marker_path}"
        fi

        if [[ -f "${marker_path}" ]]; then
            touch "${node_done}"
        else
            echo "Stage marker missing on ${node_name} after direct NFS stage" > "${fail_file}"
            return 1
        fi

        if [[ "${SLURM_NODEID}" == "0" ]]; then
            rm -f "${done_file}"
            wait_for_direct_stage "${done_file}" "${fail_file}" "${stage_state_dir}" "${nodes[@]}"
        else
            wait_for_stage_status "${done_file}" "${fail_file}"
        fi

        export ORIGINAL_DATADIR="${DATADIR}"
        export DATADIR="${stage_dir}"
        export STAGE_TO_LOCAL_NVME=0
        echo "Using staged DATADIR=${DATADIR}"
        return 0
    fi

    if [[ "${SLURM_NODEID}" == "0" ]]; then
        rm -f "${done_file}" "${fail_file}" "${seed_file}"
        local ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
        local -a staged_nodes=()
        local -a missing_nodes=()
        local node

        if [[ -f "${HOME}/.ssh/id_ed25519" ]]; then
            eval "$(ssh-agent -s)" >/dev/null
            ssh-add "${HOME}/.ssh/id_ed25519" >/dev/null 2>&1 || true
        fi

        for node in "${nodes[@]}"; do
            if [[ "${node}" == "$(hostname)" ]]; then
                if stage_is_ready_local "${stage_dir}" "${marker_path}"; then
                    staged_nodes+=("${node}")
                    sudo touch "${marker_path}" || true
                else
                    missing_nodes+=("${node}")
                fi
            elif stage_is_ready_remote "${node}" "${stage_dir}" "${marker_path}"; then
                staged_nodes+=("${node}")
                ssh "${ssh_opts[@]}" "${node}" "sudo touch '${marker_path}'" >/dev/null 2>&1 || true
            else
                missing_nodes+=("${node}")
            fi
        done

        if (( ${#staged_nodes[@]} == 0 )); then
            seed_sync_local "${src_dir}" "${stage_dir}" "${stage_workers}"
            sudo touch "${marker_path}"
            staged_nodes+=("$(hostname)")
        fi

        printf '%s\n' "${staged_nodes[0]}" > "${seed_file}"

        local -a sources=("${staged_nodes[@]}")
        local -a newly_staged=()
        local pair_count
        local idx
        local src_host
        local dst_host
        while (( ${#missing_nodes[@]} > 0 )); do
            newly_staged=()
            pair_count=${#sources[@]}
            if (( pair_count > ${#missing_nodes[@]} )); then
                pair_count=${#missing_nodes[@]}
            fi

            for (( idx=0; idx<pair_count; idx++ )); do
                src_host="${sources[idx]}"
                dst_host="${missing_nodes[idx]}"
                fanout_copy_stage "${src_host}" "${dst_host}" "${stage_dir}" "${marker_rel}" &
            done
            wait

            for (( idx=0; idx<pair_count; idx++ )); do
                newly_staged+=("${missing_nodes[idx]}")
            done
            missing_nodes=("${missing_nodes[@]:pair_count}")
            sources+=("${newly_staged[@]}")
        done

        for node in "${nodes[@]}"; do
            if [[ "${node}" == "$(hostname)" ]]; then
                stage_is_ready_local "${stage_dir}" "${marker_path}" || { echo "Missing local stage after fan-out" > "${fail_file}"; return 1; }
            elif ! stage_is_ready_remote "${node}" "${stage_dir}" "${marker_path}"; then
                echo "Stage marker missing on ${node}" > "${fail_file}"
                return 1
            fi
        done

        touch "${done_file}"
    else
        wait_for_stage_status "${done_file}" "${fail_file}"
    fi

    stage_is_ready_local "${stage_dir}" "${marker_path}" || wait_for_stage_status "${done_file}" "${fail_file}"

    export ORIGINAL_DATADIR="${DATADIR}"
    export DATADIR="${stage_dir}"
    export STAGE_TO_LOCAL_NVME=0
    echo "Using staged DATADIR=${DATADIR}"
}

stage_dataset_to_local_nvme

select_local_or_shared_datadir() {
    if [[ "${HYBRID_LOCAL_OR_NFS}" != "1" ]]; then
        return 0
    fi

    local src_dir="${DATADIR%/}"
    local local_base="${LOCAL_DATADIR_BASE%/}"
    local local_user="${USER:-${SLURM_JOB_USER:-flux1}}"
    local default_stage_dir="${local_base}/${local_user}/$(basename "${src_dir}")"
    local stage_dir="${LOCAL_DATADIR:-${default_stage_dir}}"
    local marker_path="${stage_dir}/.flux_stage_ready_v1"

    export ORIGINAL_DATADIR="${DATADIR}"
    export STAGE_TO_LOCAL_NVME=0
    if stage_is_ready_local "${stage_dir}" "${marker_path}"; then
        export DATADIR="${stage_dir}"
        echo "Using local staged DATADIR=${DATADIR} on $(hostname)"
    else
        echo "Local stage missing on $(hostname); using shared DATADIR=${DATADIR}"
    fi
}

select_local_or_shared_datadir

echo "Running on host: $(hostname)"
sudo mkdir -p "${LOGDIR}" && sudo chmod 777 "${LOGDIR}"
sudo mkdir -p "${RUNTIME_TMP_DIR}"/{tmp,triton,torchinductor,torch,hiprtc,xdg}
sudo chmod -R 777 "${RUNTIME_TMP_DIR}"

export TMPDIR="${RUNTIME_TMP_DIR}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export XDG_CACHE_HOME="${RUNTIME_TMP_DIR}/xdg"
export TRITON_CACHE_DIR="${RUNTIME_TMP_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_TMP_DIR}/torchinductor"
export TORCH_HOME="${RUNTIME_TMP_DIR}/torch"
export PYTORCH_KERNEL_CACHE_PATH="${RUNTIME_TMP_DIR}/torch"
export HIPRTC_CACHE_PATH="${RUNTIME_TMP_DIR}/hiprtc"

_cont_mounts=(
    "--volume=${DATADIR}:/dataset"
    "--volume=${LOGDIR}:/results"
    "--volume=${REPO_DIR}:/workspace/code"
    "--volume=${REPO_DIR}/utilities:/workspace/utilities"
    "--volume=${RUNTIME_TMP_DIR}:${RUNTIME_TMP_DIR}"
)

source "${_config_file}"

mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DGXSYSTEM)

readonly _sbatch_file="${CONFIG_DIR}/job_64nodes_mlperf.sbatch"
if [[ -f "${_sbatch_file}" ]]; then
    mapfile -t _sbatch_env < <(
        grep '^export ' "${_sbatch_file}" > /tmp/sbatch_exports.sh
        env -i bash -c ". /tmp/sbatch_exports.sh && compgen -e" | grep -E -v '^(PWD|SHLVL)'
        rm -f /tmp/sbatch_exports.sh
    )
else
    _sbatch_env=()
fi

mapfile -t _all_env < <(
    printf '%s\n' "${_config_env[@]}" "${_sbatch_env[@]}" |
    awk '!seen[$0]++' |
    while read -r v; do
        printf '%s\n' "--env=${v}=${!v-}"
    done
)

_passthrough_env=(
    NCCL_DEBUG
    NCCL_DEBUG_SUBSYS
    NCCL_DEBUG_FILE
    TORCH_CPP_LOG_LEVEL
    RUNTIME_TMP_DIR
    TMPDIR
    TEMP
    TMP
    XDG_CACHE_HOME
    TRITON_CACHE_DIR
    TORCHINDUCTOR_CACHE_DIR
    TORCH_HOME
    PYTORCH_KERNEL_CACHE_PATH
    HIPRTC_CACHE_PATH
)
for v in "${_passthrough_env[@]}"; do
    if [[ -n "${!v-}" ]]; then
        _all_env+=("--env=${v}=${!v}")
    fi
done

if [[ -z "${DGXNGPU:-}" ]]; then
    echo "DGXNGPU is not set after loading ${_config_file}" >&2
    exit 1
fi

if [[ -z "${DGXNNODES:-}" && -n "${SLURM_NNODES:-}" ]]; then
    export DGXNNODES="${SLURM_NNODES}"
    _all_env+=(--env=DGXNNODES="${DGXNNODES}")
fi

cleanup_docker() {
    sudo docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

sudo docker run --rm --init --detach \
      --net=host --uts=host \
      --ipc=host \
      --device /dev/dri \
      --device /dev/kfd \
      --device /dev/infiniband \
      --group-add video \
      --cap-add SYS_PTRACE \
      --cap-add IPC_LOCK \
      --privileged \
      --ulimit memlock=-1:-1 \
      --ulimit nofile="${ULIMIT_NOFILE}:${ULIMIT_NOFILE}" \
      --security-opt seccomp=unconfined \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity

sleep 5

if [[ $RUNTIME_TUNABLES == 1 ]]; then
  bash "${REPO_DIR}/runtime_tunables.sh"
fi

sudo docker exec "${_cont_name}" true
sudo docker exec "${_cont_name}" bash -lc 'echo "Container open files limit on $(hostname): $(ulimit -n)"'

_domain=$(hostname -d 2>/dev/null || true)
_STEP_NODELIST="${SLURM_STEP_NODELIST:-${SLURM_NODELIST:-${SLURM_JOB_NODELIST}}}"
for _node in $(scontrol show hostnames "${_STEP_NODELIST}"); do
    _node_ip=$(getent ahostsv4 "${_node}" | awk '{print $1}' | sort -u | grep -v '^127\.' | head -n 1 || true)
    if [[ -z "${_node_ip}" ]]; then
        continue
    fi
    _fqdn="${_node}"
    if [[ -n "${_domain}" ]]; then
        _fqdn="${_node}.${_domain}"
    fi
    sudo docker exec "${_cont_name}" bash -lc \
        "grep -qE '^[[:space:]]*${_node_ip}[[:space:]]+.*\\b${_node}\\b' /etc/hosts || echo '${_node_ip} ${_fqdn} ${_node}' >> /etc/hosts" || true
done

_default_if=$(ip route show default 2>/dev/null | awk '{print $5; exit}')
if [[ -z "${_default_if}" ]]; then
    _default_if=$(route 2>/dev/null | grep '^default' | grep -o '[^ ]*$' | head -n 1 || true)
fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${_default_if}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${_default_if}}"
MASTER_HOSTNAME=$(scontrol show hostnames "${_STEP_NODELIST}" | head -n 1)
_resolved_master_addr=$(getent ahostsv4 "${MASTER_HOSTNAME}" | awk '{print $1}' | sort -u | grep -v '^127\.' | head -n 1 || true)
export MASTER_ADDR="${_resolved_master_addr:-${MASTER_HOSTNAME}}"
export MASTER_PORT=29501

echo "=== Node $(hostname) starting ==="
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

for _experiment_index in $(seq 0 $((NEXP-1))); do
{
  echo "Beginning trial ${_experiment_index} of ${NEXP}"
  if [[ $CLEAR_CACHES == 1 ]]; then
    bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    sudo docker exec ${_cont_name} python -c "
from mlperf_logging import mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=True)"
  fi

  export SEED=$(($ROOT_SEED - 1 + 10#$_experiment_index))
  _all_env+=(--env=SEED=$SEED)
  printf 'launching experiment using:'
  printf ' %q' "${_all_env[@]}" "${_cont_name}" ./run_and_time.sh
  printf '\n'
  sudo docker exec \
    -e "MASTER_ADDR=$MASTER_ADDR" \
    -e "MASTER_PORT=$MASTER_PORT" \
    -e "SLURM_NNODES=$SLURM_NNODES" \
    -e "SLURM_NODEID=$SLURM_NODEID" \
    -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" \
    -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME" \
    -e "ULIMIT_NOFILE=$ULIMIT_NOFILE" \
    "${_all_env[@]}" \
    "${_cont_name}" \
    bash -lc 'ulimit -n "${ULIMIT_NOFILE}"; echo "Run open files limit on $(hostname): $(ulimit -n)"; exec bash ./run_and_time.sh'
}
done
