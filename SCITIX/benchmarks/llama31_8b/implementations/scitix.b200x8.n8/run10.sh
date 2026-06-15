#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-.}"
mkdir -p "$OUTDIR"

for i in $(seq 0 4); do
  SEED_BASE=$(( RANDOM ))
  seed=$((SEED_BASE + i))
  out="$OUTDIR/result_${i}.txt"

  echo "[$(date '+%F %T')] ===== Run $i START (SEED_BASE=$SEED_BASE, seed=$seed) =====" | tee -a "$out"

  CMD="mpirun --allow-run-as-root -np 64 --bind-to none \
    -x MASTER_ADDR=172.16.178.214 \
    -x SEED_BASE=$SEED_BASE \
    -x _experiment_index=$((i + 1)) \
    -x SEED=$seed \
    /workspace/ft-llm-shared/wrapper_8b.sh"

  echo "CMD: $CMD" | tee -a "$out"

  bash -lc "$CMD" >>"$out" 2>&1
  rc=$?

  echo "[$(date '+%F %T')] ===== Run $i END (rc=$rc) =====" | tee -a "$out"
done

echo "[$(date '+%F %T')] All 10 runs finished."
