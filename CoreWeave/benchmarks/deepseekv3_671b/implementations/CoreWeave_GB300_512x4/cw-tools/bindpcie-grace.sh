#!/bin/bash
# Grace-aware drop-in replacement for NVIDIA's /usr/local/bin/bindpcie.
#
# NVIDIA's bindpcie (Copyright (c) 2018-2022) parses lscpu for "Socket(s)" and
# "Core(s) per socket". Grace's lscpu publishes "Socket(s): -" (a literal dash)
# and "Core(s) per cluster: 72" instead, so under set -euo pipefail NVIDIA's
# bindpcie aborts silently on Grace before it can pin anything. See evidence:
# /tmp/bindpcie-evidence/ (probe job 1906) and the Step 2.6 section of
# specs/gb300_llama31_405b_runbook.md.
#
# Topology auto-detection (no hardcoded shape):
#
#   num_cpu_numas    = count of NUMA nodes that own CPUs (from numactl --hardware)
#   ranks_per_node   = LOCAL_WORLD_SIZE / SLURM_NTASKS_PER_NODE /
#                      OMPI_COMM_WORLD_LOCAL_SIZE, or BINDPCIE_GRACE_RANKS_PER_NODE
#                      override (mostly for interactive srun)
#   ranks_per_numa   = ranks_per_node / num_cpu_numas
#   local_numa       = local_rank / ranks_per_numa
#
# Verified live on GB300 NVL72 (4 GPUs/node, 2 CPU NUMA, 32 generic_initiator
# HBM nodes): num_cpu_numas=2, ranks_per_node=4, ranks_per_numa=2, mapping
# matches the original hardcoded local_rank/2. Documented test evidence in
# specs/gb300_llama31_405b_runbook.md Step 2.6.
#
# Future Grace shapes (GB200 NVL36 with 1 CPU NUMA, GB300 with custom Slurm
# partitions) work without code changes - fail-loud kicks in on the contract
# we cannot satisfy (uneven ranks_per_node / num_cpu_numas) instead of
# silently miscounting.
#
# Mirrors NVIDIA bindpcie's argument parser (--cpu, --mem, --ib) so swapping
# in/out is symmetric. Per AGENTS.md "Fail Fast, No Silent Fallbacks":
# - missing LOCAL_RANK / SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK -> exit 1.
# - missing ranks_per_node from any source -> exit 1.
# - ranks_per_node not a multiple of num_cpu_numas -> exit 1.
# - unknown --cpu / --mem / --ib value -> exit 1 with valid choices listed.
# - missing numactl when CPU or memory pinning is requested -> exit 1.
# We never silently fall through to "no pinning" when pinning was asked for.

set -euo pipefail

print_usage() {
  cat <<'EOF'
bindpcie-grace.sh [options] [--] COMMAND [ARG...]

CoreWeave Grace-aware drop-in for NVIDIA bindpcie. Auto-detects CPU NUMA layout
from `numactl --hardware` and per-node rank count from LOCAL_WORLD_SIZE /
SLURM_NTASKS_PER_NODE / OMPI_COMM_WORLD_LOCAL_SIZE, then maps each rank to
NUMA node `local_rank / (ranks_per_node / num_cpu_numas)`.

Options (mirror NVIDIA bindpcie):
    --cpu=node           bind rank to its detected CPU NUMA [default]
    --cpu=physcore       bind rank to a deterministic per-rank CPU slice
                         within its NUMA (cores_per_numa / ranks_per_numa
                         cores per rank). Selected by
                         MLPERF_BINDCMD_PROFILE=grace_per_core.
    --cpu=off            no CPU binding
    --mem=node           bind rank's memory to its detected CPU NUMA [default]
    --mem=off            no memory binding
    --ib=*               accepted and ignored on this cluster (NCCL_IB_HCA is
                         the canonical site for HCA pinning; see
                         tools/pipeline/gb300/fabric/nccl-gb300-roce-4rank-per-node.sbatch)

Required env (any of):
    LOCAL_RANK | SLURM_LOCALID | OMPI_COMM_WORLD_LOCAL_RANK         (the rank)
    LOCAL_WORLD_SIZE | SLURM_NTASKS_PER_NODE |                       (ranks/node)
    OMPI_COMM_WORLD_LOCAL_SIZE | BINDPCIE_GRACE_RANKS_PER_NODE
EOF
}

cpu_mode='node'
mem_mode='node'
# ib_mode parsed but ignored intentionally (see header comment).
ib_mode='off'
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)        print_usage; exit 0 ;;
    --cpu=*)          cpu_mode="${1#--cpu=}"; shift ;;
    --cpu)            cpu_mode="$2"; shift 2 ;;
    --mem=*)          mem_mode="${1#--mem=}"; shift ;;
    --mem)            mem_mode="$2"; shift 2 ;;
    --ib=*)           ib_mode="${1#--ib=}"; shift ;;
    --ib)             ib_mode="$2"; shift 2 ;;
    --)               shift; break ;;
    *)                break ;;
  esac
done

if [[ "$#" -lt 1 ]]; then
  echo 'ERROR: bindpcie-grace.sh requires a COMMAND to wrap' >&2
  print_usage >&2
  exit 1
fi

case "${cpu_mode}" in
  node|physcore|off) ;;
  *)
    echo "ERROR: bindpcie-grace.sh: unknown --cpu='${cpu_mode}'; valid: node, physcore, off" >&2
    exit 1
    ;;
esac

case "${mem_mode}" in
  node|off) ;;
  *)
    echo "ERROR: bindpcie-grace.sh: unknown --mem='${mem_mode}'; valid: node, off" >&2
    exit 1
    ;;
esac

# ib_mode is intentionally not validated. Even unknown values are accepted so
# operators can paste an existing NVIDIA-bindpcie command line in unchanged.

# Resolve local_rank in the same precedence order NVIDIA's bindpcie uses, but
# do NOT default to empty: any of LOCAL_RANK / SLURM_LOCALID /
# OMPI_COMM_WORLD_LOCAL_RANK must be set.
local_rank=""
for var in LOCAL_RANK SLURM_LOCALID OMPI_COMM_WORLD_LOCAL_RANK; do
  if [[ -n "${!var:-}" ]]; then
    local_rank="${!var}"
    break
  fi
done
if [[ -z "${local_rank}" ]]; then
  echo 'ERROR: bindpcie-grace.sh: cannot resolve local rank from LOCAL_RANK / SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK' >&2
  exit 1
fi
if ! [[ "${local_rank}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: bindpcie-grace.sh: local_rank='${local_rank}' is not a non-negative integer" >&2
  exit 1
fi

# Short-circuit: if the operator asked for no pinning at all, we never need
# to consult numactl topology or ranks_per_node. Just exec the wrapped command.
if [[ "${cpu_mode}" = "off" && "${mem_mode}" = "off" ]]; then
  exec "$@"
fi

# numactl is required for pinning AND for `numactl --hardware` topology
# detection below. Fail loud (per AGENTS.md) instead of silently skipping.
if ! command -v numactl >/dev/null 2>&1; then
  echo 'ERROR: bindpcie-grace.sh: numactl not on PATH but pinning was requested' >&2
  exit 1
fi

# Detect CPU NUMA count from `numactl --hardware`. We count NUMA nodes whose
# `cpus:` line lists at least one CPU; the remaining nodes are
# generic_initiator / HBM regions (Grace exposes ~32 of those) and are not
# valid pinning targets.
num_cpu_numas=$(numactl --hardware | awk '/^node [0-9]+ cpus:/ && NF>3 {n++} END {print n+0}')
if [[ "${num_cpu_numas}" -lt 1 ]]; then
  echo 'ERROR: bindpcie-grace.sh: numactl --hardware reported zero CPU-bearing NUMA nodes' >&2
  exit 1
fi

# Resolve ranks_per_node. Operator override wins so interactive `srun`
# sessions where neither LOCAL_WORLD_SIZE nor SLURM_NTASKS_PER_NODE is set
# can still get correct pinning.
ranks_per_node=""
if [[ -n "${BINDPCIE_GRACE_RANKS_PER_NODE:-}" ]]; then
  ranks_per_node="${BINDPCIE_GRACE_RANKS_PER_NODE}"
else
  for var in LOCAL_WORLD_SIZE SLURM_NTASKS_PER_NODE OMPI_COMM_WORLD_LOCAL_SIZE; do
    if [[ -n "${!var:-}" ]]; then
      ranks_per_node="${!var}"
      break
    fi
  done
fi
if [[ -z "${ranks_per_node}" ]]; then
  echo 'ERROR: bindpcie-grace.sh: cannot resolve ranks_per_node from LOCAL_WORLD_SIZE / SLURM_NTASKS_PER_NODE / OMPI_COMM_WORLD_LOCAL_SIZE / BINDPCIE_GRACE_RANKS_PER_NODE' >&2
  exit 1
fi
if ! [[ "${ranks_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: bindpcie-grace.sh: ranks_per_node='${ranks_per_node}' is not a positive integer" >&2
  exit 1
fi

if (( ranks_per_node % num_cpu_numas != 0 )); then
  echo "ERROR: bindpcie-grace.sh: ranks_per_node=${ranks_per_node} is not a multiple of num_cpu_numas=${num_cpu_numas}; cannot evenly pin ranks to NUMA nodes. Pick a different shape, set BINDPCIE_GRACE_RANKS_PER_NODE explicitly, or run with --cpu=off --mem=off." >&2
  exit 1
fi

readonly num_cpu_numas
readonly ranks_per_node
readonly ranks_per_numa=$(( ranks_per_node / num_cpu_numas ))
readonly local_numa=$(( local_rank / ranks_per_numa ))

if (( local_numa >= num_cpu_numas )); then
  max_rank=$(( ranks_per_node - 1 ))
  echo "ERROR: bindpcie-grace.sh: computed local_numa=${local_numa} >= num_cpu_numas=${num_cpu_numas} for local_rank=${local_rank} (ranks_per_numa=${ranks_per_numa}); local_rank must be in [0, ${max_rank}]" >&2
  exit 1
fi

# Topology drift check. The arithmetic above (local_rank / ranks_per_numa)
# only lands the right rank on the right NUMA when the GPU-to-NUMA layout is
# monotonic on this SKU (i.e. GPU N sits on NUMA `N / ranks_per_numa`). Today
# on GB300 NVL72 it is - GPU0,1 -> NUMA0; GPU2,3 -> NUMA1, captured in
# docs/grace-topology-evidence.md from `nvidia-smi topo -m`. A future SKU
# that ships an alternating layout (GPU0,2 -> NUMA0; GPU1,3 -> NUMA1) would
# silently mis-pin every odd rank with this arithmetic, paying the cross-
# socket coherence penalty for every host<->device transfer. Per AGENTS.md
# "Fail Fast, No Silent Fallbacks" we verify against the GPU's actual NUMA
# Affinity column from `nvidia-smi topo -m` and exit 1 on mismatch instead.
#
# Gates:
#   - BINDPCIE_GRACE_VERIFY_TOPO=0 disables the check (dev boxes without
#     nvidia-smi, smoke jobs that intentionally use a stale topology).
#     Default is 1; production runs always validate.
#   - `command -v nvidia-smi`: silently skip when nvidia-smi is not on PATH
#     (e.g. macOS unit-test boxes), so non-cluster invocations still work.
#   - Empty actual_numa from the awk parser (no matching GPU row): silently
#     skip rather than fail-loud, so a future `nvidia-smi topo -m` format
#     change does not block jobs while we investigate.
if [[ "${BINDPCIE_GRACE_VERIFY_TOPO:-1}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  # Resolve the GPU index this rank owns. The canonical answer is local_rank
  # because bindpcie runs BEFORE the framework selects its GPU - the
  # cluster's per-rank GPU allocation is implicit in local_rank.
  #
  # CUDA_VISIBLE_DEVICES is honored only as an explicit single-GPU override
  # (a single integer with no commas). On many clusters (CoreWeave GB300
  # NVL72 included; live evidence: slurm job 3455 task 7) Slurm sets
  # CUDA_VISIBLE_DEVICES="0,1,2,3" globally so every rank sees all 4 GPUs,
  # leaving the framework to pick its device by LOCAL_RANK. Naively taking
  # the first entry of that comma-list would always look up GPU0 regardless
  # of rank and produce a false-positive drift trip on every rank > 0.
  if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
    gpu_idx="${CUDA_VISIBLE_DEVICES}"
  else
    gpu_idx="${local_rank}"
  fi
  if [[ "${gpu_idx}" =~ ^[0-9]+$ ]]; then
    # Walk the matching `GPU<idx>` row from the right and pick the last
    # numeric token. `nvidia-smi topo -m` ends each row with `CPU Affinity`,
    # `NUMA Affinity`, and `GPU NUMA ID`. On Grace the trailing `GPU NUMA ID`
    # is `N/A` (non-numeric), so the rightmost numeric token is the NUMA
    # Affinity column - exactly what we want.
    actual_numa=$(nvidia-smi topo -m 2>/dev/null \
      | awk -v g="GPU${gpu_idx}" '$1 == g { for (i=NF; i>=1; i--) if ($i ~ /^[0-9]+$/) { print $i; exit } }') || true
    if [[ -n "${actual_numa}" && "${actual_numa}" != "${local_numa}" ]]; then
      cat >&2 <<EOF
ERROR: bindpcie-grace.sh: topology drift detected.
Derived local_numa=${local_numa} from local_rank=${local_rank} arithmetic
(ranks_per_numa=${ranks_per_numa}, num_cpu_numas=${num_cpu_numas}), but
GPU${gpu_idx} reports NUMA Affinity=${actual_numa} via 'nvidia-smi topo -m'.
Either the GPU-to-NUMA layout is no longer monotonic on this SKU (compare
against docs/grace-topology-evidence.md) or the local_rank-to-GPU mapping
changed. Refusing to mis-pin.
Set BINDPCIE_GRACE_VERIFY_TOPO=0 to bypass while you investigate.
EOF
      exit 1
    fi
  fi
fi

# Per-core slice derivation. Only needed for --cpu=physcore; we still compute
# it eagerly so the audit log line below (which advertises cpu_bind=cpus=...)
# has a value. Parses the actual CPU list from `numactl --hardware` for our
# local_numa rather than assuming contiguous ranges, so non-contiguous Grace
# SKUs work without code changes - on today's GB300 NVL72 the list happens
# to be 0..71 / 72..143 (contiguous; see docs/grace-topology-evidence.md).
phys_cpu_spec=""
if [[ "${cpu_mode}" == "physcore" ]]; then
  numa_cpus_str=$(numactl --hardware \
    | awk -v n="${local_numa}" '$1 == "node" && $2 == n && $3 == "cpus:" { for (i=4; i<=NF; i++) printf "%s%s", (i>4?",":""), $i }')
  IFS=',' read -r -a numa_cpus_arr <<<"${numa_cpus_str}"
  cores_per_numa="${#numa_cpus_arr[@]}"
  if (( cores_per_numa < 1 )); then
    echo "ERROR: bindpcie-grace.sh: numactl --hardware reported zero CPUs for NUMA ${local_numa}" >&2
    exit 1
  fi
  if [[ "${BINDPCIE_GRACE_PHYSCORE_FULL_NUMA:-0}" == "1" ]]; then
    # Bisection knob: every rank on a NUMA pins to the FULL NUMA cpu list
    # (ranks on the same NUMA overlap across all cores). Used to test
    # whether the NCCL ncclSystemError observed in cluster job 3464 is
    # caused by mask narrowness (36 cores per rank on GB300 NVL72) or by
    # `--physcpubind` syntax itself. Always-passes the slicing gate
    # because it does no slicing. See specs/gb300_llama31_405b_runbook.md
    # Step 2.6 "Cluster experiment outcome (job 3455 / 3464)".
    #
    # FULL_NUMA wins over BINDPCIE_GRACE_PHYSCORE_WIDTH when both env
    # knobs are set (no surprise: FULL_NUMA is just WIDTH=cores_per_numa).
    phys_cpu_spec="${numa_cpus_str}"
  elif [[ -n "${BINDPCIE_GRACE_PHYSCORE_WIDTH:-}" ]]; then
    # Mask-width bisection knob: every rank on a NUMA pins to the FIRST N
    # cores of its NUMA cpu list (all ranks on the NUMA share the window;
    # ranks overlap). Used to find the minimum viable per-rank cpuset for
    # NCCL+IB on Spectrum-X+GB300 between the known-broken default
    # cores_per_rank=36 (job 3464) and the known-working FULL_NUMA=72
    # (job 3915). See specs/gb300_llama31_405b_runbook.md Step 2.6
    # "Bisection knobs (for diagnosing physcore failures)".
    #
    # Precedence with FULL_NUMA: FULL_NUMA wins (handled above; this
    # branch is only reached when FULL_NUMA is unset/0).
    if ! [[ "${BINDPCIE_GRACE_PHYSCORE_WIDTH}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: bindpcie-grace.sh: BINDPCIE_GRACE_PHYSCORE_WIDTH='${BINDPCIE_GRACE_PHYSCORE_WIDTH}' is not a positive integer" >&2
      exit 1
    fi
    if (( BINDPCIE_GRACE_PHYSCORE_WIDTH > cores_per_numa )); then
      echo "ERROR: bindpcie-grace.sh: BINDPCIE_GRACE_PHYSCORE_WIDTH=${BINDPCIE_GRACE_PHYSCORE_WIDTH} > cores_per_numa=${cores_per_numa}; cap the WIDTH at cores_per_numa or use BINDPCIE_GRACE_PHYSCORE_FULL_NUMA=1 (which equals WIDTH=cores_per_numa)" >&2
      exit 1
    fi
    for (( _i=0; _i<BINDPCIE_GRACE_PHYSCORE_WIDTH; _i++ )); do
      phys_cpu_spec+="${phys_cpu_spec:+,}${numa_cpus_arr[_i]}"
    done
  else
    if (( cores_per_numa % ranks_per_numa != 0 )); then
      echo "ERROR: bindpcie-grace.sh: cores_per_numa=${cores_per_numa} (NUMA ${local_numa}) is not a multiple of ranks_per_numa=${ranks_per_numa}; cannot evenly slice cores per rank for --cpu=physcore. Pick a different shape, set BINDPCIE_GRACE_RANKS_PER_NODE explicitly, set BINDPCIE_GRACE_PHYSCORE_FULL_NUMA=1 to overlap ranks on the full NUMA, set BINDPCIE_GRACE_PHYSCORE_WIDTH=N to overlap ranks on a specific N-core window, or use --cpu=node." >&2
      exit 1
    fi
    cores_per_rank=$(( cores_per_numa / ranks_per_numa ))
    rank_within_numa=$(( local_rank % ranks_per_numa ))
    core_lo_idx=$(( rank_within_numa * cores_per_rank ))
    core_hi_idx=$(( core_lo_idx + cores_per_rank - 1 ))
    for (( _i=core_lo_idx; _i<=core_hi_idx; _i++ )); do
      phys_cpu_spec+="${phys_cpu_spec:+,}${numa_cpus_arr[_i]}"
    done
  fi
  if [[ -z "${phys_cpu_spec}" ]]; then
    echo "ERROR: bindpcie-grace.sh: empty phys_cpu_spec for NUMA ${local_numa} (cores_per_numa=${cores_per_numa}, ranks_per_numa=${ranks_per_numa}, full_numa=${BINDPCIE_GRACE_PHYSCORE_FULL_NUMA:-0}, width=${BINDPCIE_GRACE_PHYSCORE_WIDTH:-unset})" >&2
    exit 1
  fi
fi

declare -a numactl_args=()
case "${cpu_mode}" in
  node)     numactl_args+=( "--cpunodebind=${local_numa}" ) ;;
  physcore) numactl_args+=( "--physcpubind=${phys_cpu_spec}" ) ;;
  off)      ;;
esac
case "${mem_mode}" in
  node) numactl_args+=( "--membind=${local_numa}" ) ;;
  off)  ;;
esac

# Per-rank pinning audit log line. One line per rank per job, written to
# stderr just before exec so it is captured by the slurm log alongside the
# rank's own output. Schema is stable so
# `gb300_mlperf.sh summarize-pinning <slurm_log>` can parse it post-job.
# Default-on; BINDPCIE_GRACE_LOG_PINNING=0 disables (operator hatch for
# downstream parsers that cannot tolerate the extra line).
if [[ "${BINDPCIE_GRACE_LOG_PINNING:-1}" == "1" ]]; then
  case "${cpu_mode}" in
    node)     _log_cpu_bind="numa=${local_numa}" ;;
    physcore) _log_cpu_bind="cpus=${phys_cpu_spec}" ;;
    off)      _log_cpu_bind="off" ;;
  esac
  case "${mem_mode}" in
    node) _log_mem_bind="${local_numa}" ;;
    off)  _log_mem_bind="off" ;;
  esac
  printf 'bindpcie-grace: rank=%d local_numa=%d ranks_per_numa=%d num_cpu_numas=%d cpu_bind=%s mem_bind=%s\n' \
    "${local_rank}" "${local_numa}" "${ranks_per_numa}" "${num_cpu_numas}" \
    "${_log_cpu_bind}" "${_log_mem_bind}" >&2
fi

[[ "${DEBUG:-0}" = "1" ]] && set -x
exec numactl "${numactl_args[@]}" -- "$@"
