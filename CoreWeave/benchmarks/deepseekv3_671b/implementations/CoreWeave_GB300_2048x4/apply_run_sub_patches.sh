#!/bin/bash
# Idempotent patcher for the NVIDIA-shipped run.sub.
#
#   1. Drop  --container-env=MASTER_PORT,MASTER_ADDR,NCCL_SHARP_GROUP_SIZE_THRESH,NCCL_NVLS_ENABLE
#      from the training srun (Hydra needs more than the four-name allowlist).
#   2. Replace the training srun's `./run_and_time.sh` with
#      `/results/run_wrapper.sh`.
#   3. Rewrite the in-training compliance output filename from
#      compliance_${DATESTAMP}.out to
#      compliance_${DATESTAMP}_${_experiment_index}.out (both occurrences)
#      so trial 1-4 results aren't clobbered by trial 5.
#   4. Insert `export LAUNCH_DEADLINE=...` at the top of the NEXP for-loop
#      so run_wrapper.sh can sync rank-startup to a shared deadline
#      (LAUNCH_SYNC_SECS, default 30). Avoids the late-rank/early-rank
#      drift that causes NCCL bootstrap timeouts at scale.

set -euo pipefail

target="${1:-run.sub}"
[[ -f "$target" ]] || { echo "ERROR: $target not found" >&2; exit 1; }

if grep -q '"slurm2pytorch" /results/run_wrapper.sh' "$target" \
   && grep -qF 'compliance_${DATESTAMP}_${_experiment_index}.out' "$target" \
   && grep -qF 'export LAUNCH_DEADLINE=' "$target"; then
  echo "apply_run_sub_patches: $target already patched - no change."
  exit 0
fi

backup="${target}.preMyPatch"
[[ -f "$backup" ]] || cp -p "$target" "$backup"

sed -i 's|--container-env=MASTER_PORT,MASTER_ADDR,NCCL_SHARP_GROUP_SIZE_THRESH,NCCL_NVLS_ENABLE||' "$target"
sed -i 's|"slurm2pytorch" \./run_and_time\.sh|"slurm2pytorch" /results/run_wrapper.sh|' "$target"
sed -i 's|compliance_\${DATESTAMP}\.out|compliance_\${DATESTAMP}_\${_experiment_index}.out|g' "$target"
sed -i '/^        echo "RUNANDTIME_START/i\        export LAUNCH_DEADLINE=$(( $(date +%s) + ${LAUNCH_SYNC_SECS:-30} ))' "$target"

if ! grep -q '"slurm2pytorch" /results/run_wrapper.sh' "$target"; then
  echo "ERROR: patch 2 (wrapper redirect) did not apply." >&2
  cp -p "$backup" "$target"
  exit 1
fi
if grep -q -- '--container-env=MASTER_PORT,MASTER_ADDR,NCCL_SHARP_GROUP_SIZE_THRESH,NCCL_NVLS_ENABLE' "$target"; then
  echo "ERROR: patch 1 (drop --container-env) did not apply." >&2
  cp -p "$backup" "$target"
  exit 1
fi
if ! grep -qF 'compliance_${DATESTAMP}_${_experiment_index}.out' "$target"; then
  echo "ERROR: patch 3 (per-trial compliance) did not apply." >&2
  cp -p "$backup" "$target"
  exit 1
fi
if ! grep -qF 'export LAUNCH_DEADLINE=' "$target"; then
  echo "ERROR: patch 4 (LAUNCH_DEADLINE insert) did not apply." >&2
  cp -p "$backup" "$target"
  exit 1
fi

echo "apply_run_sub_patches: patched $target (backup at $backup)"
