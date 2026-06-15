#!/bin/bash
# Launcher for DeepSeek v3 671B training on CoreWeave GB300.
#
# Sources the chosen config (one of the supplementary 64/128/256/480-node
# variants), sets RoCE/Spectrum-X NCCL env, stages config_*.sh + run_wrapper.sh
# into LOGDIR (so they appear at /results/ in-container), then sbatches run.sub.
#
# Usage:
#   ./submit_train.sh --mode=smoke
#   ./submit_train.sh --mode=submission --config=config_GB300_128x4x120xtp1pp4ep32cp1_mxfp8_full_cg.sh
#   ./submit_train.sh --mode=nccl

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_train.sh [--mode=MODE] [--config=PATH] [--reservation=NAME]
                       [--partition=NAME] [--max-steps=N] [--after=JOBID]
                       [--exclude=NODELIST] [--dry-run]

  --mode=nccl         RUN_ONLY_NCCL=1, NEXP=1, --time=30
  --mode=smoke        NEXP=1, MAX_STEPS=5, NCCL_TEST=1     (default)
  --mode=full         NEXP=1, NCCL_TEST=0                  (skip NCCL probe)
  --mode=submission   NEXP=5, NCCL_TEST=1                  (full MLPerf cycle)

  --config=PATH       config_*.sh to source (default: 128-node)
  --reservation=NAME  Slurm reservation (default: $SBATCH_RESERVATION)
  --partition=NAME    Slurm partition (default: hpc-high)
  --max-steps=N       Override MAX_STEPS (smoke mode default 5; otherwise unset)
  --after=JOBID       Run after JOBID finishes (--dependency=afterany:JOBID).
  --exclude=NODELIST  Exclude these nodes from the allocation (sbatch --exclude).
  --dry-run           Print the sbatch command without submitting

Env overrides:
  STAGE_DATA=0/1      Cache dataset to node-local NVMe before training
                      (default 1 for smoke/full/submission, 0 for nccl).
                      When 1, sets SLOW_DATADIR=<original DATADIR> and
                      DATADIR=/tmp/deepseekv3-671b-cache; run.sub's built-in
                      per-node rsync seeds the cache before training.
  NCCL_NET_PLUGIN     NCCL network transport plugin (default `none`; try
                      `spcx` for the Spectrum-X plugin).
  NCCL_NVLS_ENABLE    Override NVLS on/off (config_common.sh sets it to 0;
                      set to 1 to flip on for an experiment).
  LAUNCH_SYNC_SECS    Seconds of rank-startup sync window (default 30). All
                      ranks of a training srun wait until a shared deadline
                      computed at trial start before proceeding to NCCL init,
                      so early starters don't time out waiting for late ones.
  DATA_NUM_WORKERS    Dataloader worker count (default 8).
  NVTX_FLAG=0/1       Wrap python with `nsys profile`. Pair with PROFILE=1.
  PROFILE=0/1         Drive cudaProfilerStart/Stop in python at
                      PROFILE_START_STEP / PROFILE_END_STEP (defaults 10/12).
                      Both NVTX_FLAG and PROFILE must be 1 for a useful
                      capture. .nsys-rep files land in LOGDIR.
  BREAKDOWN=0/1       Add NVTX ranges in NeMo+MCore for finer-grained
                      profile breakdown (default 0).
  PROFILE_RANKS       Comma-separated ranks to capture (default `0`).
  MLPERF_SUBMISSION_ORG       Org embedded in MLLOG (default CoreWeave).
  MLPERF_SUBMISSION_PLATFORM  Platform suffix; emitted as <NODES>x<value>
                              (default GB300_NVL72).
EOF
}

MODE=smoke
CONFIG="config_GB300_128x4x120xtp1pp4ep32cp1_mxfp8_full_cg.sh"
RESERVATION="${SBATCH_RESERVATION:-}"
PARTITION="hpc-high"
MAX_STEPS_OVERRIDE=""
AFTER_JOB=""
EXCLUDE_NODES=""
DRY=0

for arg in "$@"; do
  case "$arg" in
    --mode=*)         MODE="${arg#*=}" ;;
    --config=*)       CONFIG="${arg#*=}" ;;
    --reservation=*)  RESERVATION="${arg#*=}" ;;
    --partition=*)    PARTITION="${arg#*=}" ;;
    --max-steps=*)    MAX_STEPS_OVERRIDE="${arg#*=}" ;;
    --after=*)        AFTER_JOB="${arg#*=}" ;;
    --exclude=*)      EXCLUDE_NODES="${arg#*=}" ;;
    --dry-run)        DRY=1 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$RESERVATION" ]] || { echo "ERROR: --reservation required (or export SBATCH_RESERVATION)"; exit 2; }

WORKDIR="${WORKDIR:-/mnt/data/mlperf6/scratch/$USER/deepseekv3-671b-train}"
cd "$WORKDIR"

[[ -f "$CONFIG" ]] || { echo "ERROR: config '$CONFIG' not found in $WORKDIR"; exit 1; }
[[ -f run.sub ]] || { echo "ERROR: run.sub not found"; exit 1; }
[[ -f run_wrapper.sh ]] || { echo "ERROR: run_wrapper.sh not found"; exit 1; }

for marker in \
    '"slurm2pytorch" /results/run_wrapper.sh' \
    'compliance_${DATESTAMP}_${_experiment_index}.out' \
    'export LAUNCH_DEADLINE='; do
  if ! grep -qF "$marker" run.sub; then
    echo "ERROR: run.sub is missing patch marker: $marker" >&2
    echo "       Run ./apply_run_sub_patches.sh run.sub first." >&2
    exit 1
  fi
done

# Capture env overrides that need to win over values set by the upstream
# config (sourcing config_common.sh resets them). Restored after source.
NCCL_NVLS_ENABLE_OVERRIDE="${NCCL_NVLS_ENABLE:-}"

# shellcheck source=/dev/null
source "$CONFIG"

if [[ -n "$NCCL_NVLS_ENABLE_OVERRIDE" ]]; then
  export NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE_OVERRIDE"
fi

case "$MODE" in
  nccl)
    export NEXP=1 NCCL_TEST=1 RUN_ONLY_NCCL=1
    export WALLTIME=30
    SBATCH_TIME=$WALLTIME
    : "${STAGE_DATA:=0}"
    ;;
  smoke)
    export NEXP=1 NCCL_TEST=1 RUN_ONLY_NCCL=0
    : "${MAX_STEPS_OVERRIDE:=5}"
    export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))
    SBATCH_TIME=$WALLTIME
    : "${STAGE_DATA:=1}"
    ;;
  full)
    export NEXP=1 NCCL_TEST=0 RUN_ONLY_NCCL=0
    export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))
    SBATCH_TIME=$WALLTIME
    : "${STAGE_DATA:=1}"
    ;;
  submission)
    export NEXP=5 NCCL_TEST=1 RUN_ONLY_NCCL=0
    export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))
    SBATCH_TIME=$WALLTIME
    : "${STAGE_DATA:=1}"
    ;;
  *) echo "Unknown mode: $MODE" >&2; usage; exit 2 ;;
esac

export MAX_STEPS_OVERRIDE STAGE_DATA

# Cluster paths
export CONT="${CONT:-/mnt/data/images/deepseekv3_671b-arm-20260511-hpcx250.sqsh}"
export DATADIR="${DATADIR:-/mnt/data/deepseekv3-671b/dataset}"
# When STAGE_DATA=1, point DATADIR at node-local NVMe and let run.sub's
# built-in SLOW_DATADIR rsync seed it before the training srun. The
# in-container /preproc_data bind then reads from /tmp instead of NFS.
# rsync skip-on-mtime makes subsequent jobs in the same pod near-no-ops.
if [[ "$STAGE_DATA" == "1" ]]; then
  export SLOW_DATADIR="$DATADIR"
  export DATADIR=/tmp/deepseekv3-671b-cache
fi
export HF_DIR="${HF_DIR:-/mnt/data/deepseekv3-671b/hf_home}"
export LOAD_CHECKPOINTS_DIR="${LOAD_CHECKPOINTS_DIR:-/mnt/data/deepseekv3-671b/checkpoint/megatron_ckpt}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/checkpoints}"
export LOGDIR="${LOGDIR:-$WORKDIR/logs}"
mkdir -p "$LOGDIR"

# Mount the CoreWeave cw-tools directory at /cw-tools in-container so the
# Grace-aware bindpcie drop-in is visible to run_and_time.sh.
CWTOOLS_HOST="${CWTOOLS_HOST:-/mnt/data/mlperf6/scratch/$USER/cw-tools}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}${CWTOOLS_HOST}:/cw-tools:ro"

# MLPerf submission metadata embedded in the MLLOG via mlperf_common.logging
# (pretrain.py:92). Should match the "submitter" / system_name values in
# system_description.json.
export MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-CoreWeave}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-GB300_NVL72}"

# RoCE / Spectrum-X NCCL env (mandatory on this fabric).
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_IB_HCA=ibp
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_ADDR_FAMILY=AF_INET6
export NCCL_IB_ADDR_RANGE=fd02::/16
export NCCL_IB_TC=96
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_NET_LATENCY=50
export NVIDIA_IMEX_CHANNELS=0
export PMIX_MCA_gds='^ds12'
export NCCL_P2P_NET_CHUNKSIZE="${NCCL_P2P_NET_CHUNKSIZE-524288}"
export NCCL_NVLS_NCHANNELS="${NCCL_NVLS_NCHANNELS-24}"
export NCCL_NVLSTREE_MAX_CHUNKSIZE="${NCCL_NVLSTREE_MAX_CHUNKSIZE-262144}"
export NCCL_NVLS_CHUNKSIZE="${NCCL_NVLS_CHUNKSIZE-262144}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HYDRA_FULL_ERROR=1

# Rank-startup sync window. run.sub computes LAUNCH_DEADLINE at the top of
# each NEXP iteration; run_wrapper.sh sleeps until that deadline so all
# ranks reach NCCL init within ms of each other.
export LAUNCH_SYNC_SECS="${LAUNCH_SYNC_SECS:-30}"

# Stage configs and wrapper into LOGDIR so they appear at /results/ in-container.
cp -f "$WORKDIR"/config_common.sh "$WORKDIR"/config_common_mxfp8.sh "$LOGDIR/" 2>/dev/null || true
cp -f "$WORKDIR"/config_GB300_*.sh "$LOGDIR/"
cp -f "$WORKDIR/run_wrapper.sh" "$LOGDIR/"
chmod +x "$LOGDIR/run_wrapper.sh"
export SELECTED_CONFIG="$CONFIG"

cat <<EOF
=== submit_train.sh ===
MODE             = $MODE
CONFIG           = $CONFIG
DGXSYSTEM        = $DGXSYSTEM
DGXNNODES        = $DGXNNODES (x $DGXNGPU GPUs = $((DGXNNODES * DGXNGPU)) total)
SEGMENT          = ${SEGMENT:-16}
NEXP             = $NEXP
MAX_STEPS        = ${MAX_STEPS_OVERRIDE:-${MAX_STEPS:-<from config>}}
WALLTIME         = $WALLTIME (sbatch --time=$SBATCH_TIME)
RESERVATION      = $RESERVATION
PARTITION        = $PARTITION
CONT             = $CONT
DATADIR          = $DATADIR
STAGE_DATA       = $STAGE_DATA
NCCL_NET_PLUGIN  = $NCCL_NET_PLUGIN
NCCL_NVLS_ENABLE = ${NCCL_NVLS_ENABLE:-<unset>}
LAUNCH_SYNC_SECS = $LAUNCH_SYNC_SECS
LOAD_CKPTS       = $LOAD_CHECKPOINTS_DIR
LOGDIR           = $LOGDIR
EOF

if [[ "${NVTX_FLAG:-0}" != "0" || "${PROFILE:-0}" != "0" ]]; then
  cat <<EOF
=== profiling ===
NVTX_FLAG          = ${NVTX_FLAG:-0}
PROFILE            = ${PROFILE:-0}
BREAKDOWN          = ${BREAKDOWN:-0}
PROFILE_START_STEP = ${PROFILE_START_STEP:-(default)}
PROFILE_END_STEP   = ${PROFILE_END_STEP:-(default)}
PROFILE_RANKS      = ${PROFILE_RANKS:-(default)}
EOF
fi

sbatch_cmd=(
  sbatch
  -N "$DGXNNODES"
  --partition="$PARTITION"
  --segment="${SEGMENT:-16}"
  --reservation="$RESERVATION"
  --gres=gpu:gb300:4
  --mem=0
  --time="$SBATCH_TIME"
  --no-requeue
  --job-name="ds671b-${MODE}"
  --output="$LOGDIR/slurm-%x-%j.out"
)
if [[ -n "$EXCLUDE_NODES" ]]; then
  sbatch_cmd+=( --exclude="$EXCLUDE_NODES" )
fi
if [[ -n "$AFTER_JOB" ]]; then
  sbatch_cmd+=( --dependency="afterany:$AFTER_JOB" )
fi
sbatch_cmd+=( run.sub )

if [[ "$DRY" -eq 1 ]]; then
  printf 'DRY-RUN: '
  printf '%q ' "${sbatch_cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${sbatch_cmd[@]}"
