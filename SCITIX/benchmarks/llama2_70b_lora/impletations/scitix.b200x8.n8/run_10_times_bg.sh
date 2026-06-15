#!/usr/bin/env bash
set -u
set -o pipefail

MASTER_ADDR="${MASTER_ADDR:-172.16.124.222}"
NP="${NP:-64}"
RUNS="${RUNS:-10}"

mkdir -p result logs

echo "MASTER_ADDR=${MASTER_ADDR}, NP=${NP}, RUNS=${RUNS}"
echo "Start: $(date -Is)"
echo

for i in $(seq 0 $((RUNS-1))); do
  SEED_BASE=$(( RANDOM ))
  seed=$((SEED_BASE + i))

  echo "========== Run ${i} / $((RUNS-1)) =========="
  echo "Using SEED=${seed}"

  run_log="logs/run_${i}.log"
  out_file="result/result_${i}.txt"
  status_file="result/status_${i}.txt"

  mpirun --allow-run-as-root \
    -np "${NP}" \
    --bind-to none \
    -x "SEED=${seed}" \
    -x "MASTER_ADDR=${MASTER_ADDR}" \
    /workspace/ft-llm-shared/wrapper.sh \
    > "${run_log}" 2>&1

  rc=$?

  echo "${rc}" > "${status_file}"
  grep -F ":::MLLOG" "${run_log}" > "${out_file}" || true

  echo "Run ${i} exit_code=${rc}"
  echo "Saved: ${out_file}"
  echo "Full log: ${run_log}"
  echo
done

echo "Done: $(date -Is)"
