#!/bin/bash
# In-container wrapper invoked by run.sub's training srun (replaces the
# default ./run_and_time.sh path). Two responsibilities:
#
#   1. Apply runtime patches to the (shared, --container-writable) image
#      overlay before any python imports happen. Two patches are needed
#      because nvidia_resiliency_ext v0.5.0 in this container has API
#      drift vs Megatron-LM 94c6b505:
#        - alias _get_write_results_queue -> get_write_results_queue
#        - drop the use_cached_data_structure kwarg passed at torch.py:700
#      flock guards against the 16 ranks/node racing each other.
#      Lock + patches all live on the container overlay (local FS), not
#      NFS, so flock semantics are safe here.
#
#   2. Source the config named by $SELECTED_CONFIG (from submit_train.sh)
#      so Hydra's ${oc.env:VAR,DEFAULT} resolves to our values
#      (DGXNNODES=128, etc.) rather than the placeholder defaults baked
#      into conf/*.yaml.
#
# /results is the LOGDIR mount; submit_train.sh stages config_*.sh + this
# wrapper into LOGDIR before sbatch.

set -uo pipefail

# Rank-startup sync. run.sub sets LAUNCH_DEADLINE at the top of each NEXP
# iteration; each rank sleeps to that absolute timestamp so all ranks reach
# the first NCCL collective together. Late starters skip the sleep.
if [[ -n "${LAUNCH_DEADLINE:-}" ]]; then
  remaining=$(( LAUNCH_DEADLINE - $(date +%s) ))
  if (( remaining > 0 )); then
    sleep "$remaining"
  fi
  printf '[%s] LAUNCH_SYNC done at %s\n' "$(hostname)" "$(date -Iseconds)"
fi

# 1. Runtime patches, once per node, before any python.
(
  flock -x 200
  f1=/usr/local/lib/python3.12/dist-packages/nvidia_resiliency_ext/checkpointing/async_ckpt/filesystem_async.py
  if ! grep -q "^get_write_results_queue = _get_write_results_queue" "$f1"; then
    echo "get_write_results_queue = _get_write_results_queue" >> "$f1"
  fi
  f2=/workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py
  if grep -qE 'async_writer_kwargs\["use_cached_data_structure"\]' "$f2"; then
    sed -i '/async_writer_kwargs\["use_cached_data_structure"\]/d' "$f2"
  fi
  f3=/workspace/llm/conf/deepseekv3_671b.yaml
  if grep -qE '^  num_workers: 8$' "$f3"; then
    sed -i 's|^  num_workers: 8$|  num_workers: ${oc.decode:${oc.env:DATA_NUM_WORKERS,8}}|' "$f3"
  fi
) 200>/tmp/ds671b-runtime-patches.lock

# 2. Re-source the chosen config for Hydra (it sources its parent + common files).
if [[ -z "${SELECTED_CONFIG:-}" ]]; then
  echo "WRAPPER_ERROR: SELECTED_CONFIG env var not set" >&2
  exit 1
fi
if [[ ! -f "/results/$SELECTED_CONFIG" ]]; then
  echo "WRAPPER_ERROR: /results/$SELECTED_CONFIG not found" >&2
  exit 1
fi
source "/results/$SELECTED_CONFIG"

# Per-knob overrides from submit_train.sh win over the config's defaults.
if [[ -n "${MAX_STEPS_OVERRIDE:-}" ]]; then
  export MAX_STEPS="$MAX_STEPS_OVERRIDE"
fi

: "${GPU_ARCH:=gb300}"
export GPU_ARCH

# NVIDIA's /usr/local/bin/bindpcie silently exits 1 on Grace (its lscpu parser
# trips on "Socket(s): -"). Use CoreWeave's drop-in (main:tools/gb300-mlperf/
# bindpcie-grace.sh), mounted at /cw-tools/ via EXTRA_MOUNTS.
export BINDCMD="/cw-tools/bindpcie-grace.sh --cpu=node --mem=node"
# bindpcie-grace.sh's NUMA verifier mis-fires on ranks where pyxis leaves
# CUDA_VISIBLE_DEVICES="0,1,2,3" (all visible) instead of narrowing to a
# single GPU: it picks the first CVD entry as gpu_idx, so for local_rank=2/3
# it compares GPU0's NUMA against derived local_numa=1 and aborts. Bypass
# while we send a fix upstream.
export BINDPCIE_GRACE_VERIFY_TOPO=0

export SKIP_TRAIN_CG_PARAM_AG=True

export LD_LIBRARY_PATH="/opt/hpcx/nccl_spectrum-x_plugin/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "/results/nccl-logs/${SLURM_JOB_ID:?}"
export NCCL_DEBUG_FILE="/results/nccl-logs/${SLURM_JOB_ID}/nccl-%h-%p.log"

if [[ "${RANK:-0}" -eq 0 ]]; then
  cat <<EOF
WRAPPER_DEBUG: DGXNNODES=${DGXNNODES:-?} DGXNGPU=${DGXNGPU:-?} MINIBS=${MINIBS:-?}
WRAPPER_DEBUG: TP=${TENSOR_MODEL_PARALLEL:-?} PP=${PIPELINE_MODEL_PARALLEL:-?} CP=${CONTEXT_PARALLEL:-?} EP=${EXPERT_PARALLEL:-?} VPP=${INTERLEAVED_PIPELINE:-?}
WRAPPER_DEBUG: SEGMENT=${SEGMENT:-?} PIPELINE_LAYOUT=${PIPELINE_LAYOUT:-?}
WRAPPER_DEBUG: MODEL_SIZE=${MODEL_SIZE:-?} DATASET=${DATASET:-?} FP8=${FP8:-?}
WRAPPER_DEBUG: LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-?} LOAD_CHECKPOINTS_DIR=${LOAD_CHECKPOINTS_DIR:-?}
WRAPPER DEBUG: MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CPU=${MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CPU:-?}
EOF
fi

./run_and_time.sh "$@" 2>"/results/wrapper_stderr_${RANK:-0}.log"
exit_code=$?

if [[ $exit_code -ne 0 && "${RANK:-0}" -eq 0 ]]; then
  echo "WRAPPER_NOTE: run_and_time.sh exited $exit_code"
  echo "WRAPPER_NOTE: exit 134 after run_stop is harmless cleanup-phase SIGABRT;"
  echo "WRAPPER_NOTE: grep the trial log for 'INFO - SUCCESS' from compliance_checker."
  echo "WRAPPER_STDERR (last 30 lines):"
  tail -30 /results/wrapper_stderr_0.log 2>/dev/null || true
fi

exit $exit_code
