#!/bin/bash

# Discover and mount the daemon host's Broadcom RDMA userspace libraries.
# Source this file and call:
#   bnxt_rdma_prepare DOCKER_ARGS_ARRAY DGXSYSTEM NNODES IMAGE
#
# The kernel driver stays on the physical host. This helper only appends
# read-only libibverbs/provider mounts and LD_LIBRARY_PATH to a Docker argument
# array. It intentionally does not run an RDMA smoke test.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: bnxt_rdma_overlay.sh must be sourced" >&2
    exit 2
fi

bnxt_rdma__discover_host_pair() {
    local image="$1"
    local verbs_override="$2"
    local provider_override="$3"
    local result

    # The launcher may itself be a container using the physical host's Docker
    # socket. Discover through the daemon host root, not the launcher's /usr.
    if ! result="$(
        docker run --rm --interactive \
            --mount type=bind,src=/,dst=/host,readonly \
            --entrypoint /bin/bash "${image}" -s -- \
            "${verbs_override}" "${provider_override}" <<'HELPER_EOF'
set -euo pipefail
exec chroot /host /bin/bash -s -- "$@" <<'HOST_EOF'
set -euo pipefail

verbs_override="$1"
provider_override="$2"

die() {
    echo "[rdma] ERROR: $*" >&2
    exit 1
}

safe_path() {
    local path="$1"
    [[ "${path}" == /* ]] || die "path must be absolute: ${path}"
    [[ "${path}" != *$'\n'* && "${path}" != *$'\t'* && "${path}" != *,* ]] \
        || die "path cannot contain newline, tab, or comma: ${path}"
}

shopt -s nullglob
hcas=(/sys/class/infiniband/bnxt_re*)
if (( ${#hcas[@]} == 0 )); then
    [[ -z "${verbs_override}${provider_override}" ]] \
        || die "explicit bnxt_re paths were supplied, but no bnxt_re HCA exists"
    echo NONE
    exit 0
fi

driver_version="$(cat /sys/module/bnxt_re/version 2>/dev/null || true)"
[[ -n "${driver_version}" ]] || die "cannot read the bnxt_re kernel driver version"

for tool in ldconfig readlink readelf strings find; do
    command -v "${tool}" >/dev/null 2>&1 || die "host is missing ${tool}"
done

if [[ -n "${verbs_override}" ]]; then
    safe_path "${verbs_override}"
    safe_path "${provider_override}"
    verbs="$(readlink -f -- "${verbs_override}" 2>/dev/null || true)"
    providers=("${provider_override}")
else
    verbs="$(ldconfig -p 2>/dev/null \
        | awk '$1 == "libibverbs.so.1" {print $NF; exit}')"
    verbs="$(readlink -f -- "${verbs}" 2>/dev/null || true)"
    providers=()
    for root in /usr/local/lib /usr/local/lib64 /usr/lib /usr/lib64 /lib /lib64; do
        [[ -e "${root}" ]] || continue
        while IFS= read -r -d '' candidate; do
            [[ "${candidate,,}" == *inbox* ]] || providers+=("${candidate}")
        done < <(find -H "${root}" -maxdepth 4 \( -type f -o -type l \) \
            -name 'libbnxt_re-rdmav*.so*' -print0 2>/dev/null)
    done
fi

[[ -f "${verbs}" && -r "${verbs}" ]] || die "cannot find the active host libibverbs.so.1"
[[ "$(readelf -d "${verbs}" 2>/dev/null \
    | sed -n 's/.*(SONAME).*\[\([^]]*\)\].*/\1/p')" == libibverbs.so.1 ]] \
    || die "active libibverbs has an unexpected SONAME: ${verbs}"

declare -A seen=()
valid=()
for candidate in "${providers[@]}"; do
    safe_path "${candidate}"
    provider="$(readlink -f -- "${candidate}" 2>/dev/null || true)"
    [[ -f "${provider}" && -r "${provider}" ]] || continue
    [[ -z "${seen[${provider}]+x}" ]] || continue
    seen["${provider}"]=1

    soname="$(readelf -d "${provider}" 2>/dev/null \
        | sed -n 's/.*(SONAME).*\[\([^]]*\)\].*/\1/p')"
    [[ "${soname}" =~ ^libbnxt_re-rdmav([0-9]+)\.so$ ]] || continue
    private_abi="${BASH_REMATCH[1]}"

    strings -a "${provider}" | grep -Fx "${driver_version}" >/dev/null || continue
    readelf --version-info "${provider}" 2>/dev/null \
        | grep -F "IBVERBS_PRIVATE_${private_abi}" >/dev/null || continue
    readelf --version-info "${verbs}" 2>/dev/null \
        | grep -F "IBVERBS_PRIVATE_${private_abi}" >/dev/null || continue

    mapfile -t templates < <(
        strings -a "${verbs}" \
            | grep -E "^/.*/lib%s-rdmav${private_abi}\\.so$" \
            | sort -u || true
    )
    (( ${#templates[@]} == 1 )) || continue
    provider_dst="${templates[0]/\%s/bnxt_re}"
    [[ "${provider_dst##*/}" == "${soname}" ]] || continue

    valid+=("${provider}"$'\t'"${soname}"$'\t'"${provider_dst}")
done

(( ${#valid[@]} == 1 )) || {
    echo "[rdma] ERROR: expected one host provider matching bnxt_re ${driver_version}; found ${#valid[@]}" >&2
    echo "[rdma] Set both BNXT_RDMA_LIBIBVERBS_HOST_PATH and BNXT_RDMA_PROVIDER_HOST_PATH to override." >&2
    exit 1
}

IFS=$'\t' read -r provider soname provider_dst <<< "${valid[0]}"
printf 'BNXT\t%s\t%s\t%s\t%s\t%s\n' \
    "${driver_version}" "${verbs}" "${provider}" "${soname}" "${provider_dst}"
HOST_EOF
HELPER_EOF
    )"; then
        echo "[rdma] ERROR: daemon-host Broadcom RDMA discovery failed" >&2
        return 1
    fi

    printf '%s' "${result}"
}

bnxt_rdma_prepare() {
    if (( $# != 4 )) || [[ ! "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "[rdma] ERROR: usage: bnxt_rdma_prepare ARRAY DGXSYSTEM NNODES IMAGE" >&2
        return 2
    fi

    local -n docker_args="$1"
    local platform="$2"
    local nnodes="$3"
    local image="$4"
    local new_verbs old_verbs new_provider old_provider verbs_override provider_override
    local discovery mode driver verbs provider soname provider_dst extra
    local image_env image_ld_path line

    [[ "${nnodes}" =~ ^[1-9][0-9]*$ ]] || {
        echo "[rdma] ERROR: NNODES must be a positive integer: ${nnodes}" >&2
        return 1
    }
    [[ "${platform}" == MI* && "${nnodes}" -gt 1 ]] || return 0

    if [[ "${NCCL_IB_DISABLE-}" == 1 ]]; then
        echo "[rdma] NCCL_IB_DISABLE=1: skipping Broadcom userspace mounts"
        return 0
    fi

    new_verbs="${BNXT_RDMA_LIBIBVERBS_HOST_PATH-}"
    old_verbs="${MLPERF_HOST_LIBIBVERBS_PATH-}"
    new_provider="${BNXT_RDMA_PROVIDER_HOST_PATH-}"
    old_provider="${MLPERF_HOST_BNXT_PROVIDER_PATH-}"
    if [[ -n "${new_verbs}" && -n "${old_verbs}" && "${new_verbs}" != "${old_verbs}" ]] \
        || [[ -n "${new_provider}" && -n "${old_provider}" && "${new_provider}" != "${old_provider}" ]]; then
        echo "[rdma] ERROR: BNXT_RDMA_* paths conflict with their MLPERF_HOST_* aliases" >&2
        return 1
    fi
    verbs_override="${new_verbs:-${old_verbs}}"
    provider_override="${new_provider:-${old_provider}}"
    if [[ -n "${verbs_override}" && -z "${provider_override}" ]] \
        || [[ -z "${verbs_override}" && -n "${provider_override}" ]]; then
        echo "[rdma] ERROR: libibverbs and provider overrides must be set together" >&2
        return 1
    fi

    discovery="$(bnxt_rdma__discover_host_pair \
        "${image}" "${verbs_override}" "${provider_override}")" || return 1
    if [[ "${discovery}" == NONE ]]; then
        echo "[rdma] no bnxt_re HCA on the daemon host; no extra mounts needed"
        return 0
    fi

    extra=""
    IFS=$'\t' read -r mode driver verbs provider soname provider_dst extra <<< "${discovery}"
    if [[ "${mode}" != BNXT || "${verbs}" != /* || "${provider}" != /* \
        || ! "${soname}" =~ ^libbnxt_re-rdmav[0-9]+\.so$ \
        || "${provider_dst}" != /* || -n "${extra}" ]]; then
        echo "[rdma] ERROR: malformed daemon-host discovery result" >&2
        return 1
    fi

    image_env="$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${image}")"
    image_ld_path=""
    while IFS= read -r line; do
        [[ "${line}" == LD_LIBRARY_PATH=* ]] && image_ld_path="${line#LD_LIBRARY_PATH=}"
    done <<< "${image_env}"

    docker_args+=(
        --mount "type=bind,src=${verbs},dst=/opt/host-rdma/lib/libibverbs.so.1,readonly"
        --mount "type=bind,src=${verbs},dst=/opt/host-rdma/lib/libibverbs.so,readonly"
        --mount "type=bind,src=${provider},dst=${provider_dst},readonly"
        --env "LD_LIBRARY_PATH=/opt/host-rdma/lib${image_ld_path:+:${image_ld_path}}"
    )

    echo "[rdma] bnxt_re kernel=${driver}"
    echo "[rdma] mount libibverbs=${verbs}"
    echo "[rdma] mount provider=${provider} -> ${provider_dst}"
}
