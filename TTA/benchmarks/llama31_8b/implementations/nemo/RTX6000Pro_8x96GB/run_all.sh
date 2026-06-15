#!/bin/bash
# MLPerf Training v6.0 - LLaMA 3.1 8B
# Run all 10 required runs for Closed Division submission

set -e

WORKSPACE=${1:-/home/training/mlperf_workspace}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_RUNS=10

echo "============================================"
echo " MLPerf Training v6.0 - Full Submission"
echo " 10 runs for Closed Division"
echo "============================================"

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo ">>> Starting Run $i / $NUM_RUNS <<<"
    bash "$SCRIPT_DIR/run_and_time.sh" "$WORKSPACE" "$i"
    echo ">>> Run $i complete <<<"
done

echo ""
echo "=== TTT Summary ==="
RESULT_DIR="$WORKSPACE/final_submission/TTA/results/RTX6000Pro_8x96GB/llama31_8b"
for f in "$RESULT_DIR"/result_*.txt; do
    RUN=$(basename "$f")
    START=$(grep "run_start" "$f" | head -1 | grep -oP '"time_ms": \K[0-9]+')
    STOP=$(grep '"status": "success"' "$f" | grep "run_stop" | head -1 | grep -oP '"time_ms": \K[0-9]+')
    if [ -n "$START" ] && [ -n "$STOP" ]; then
        TTT_MIN=$(echo "scale=2; ($STOP - $START) / 60000" | bc)
        echo "  $RUN: ${TTT_MIN} min"
    else
        echo "  $RUN: FAILED or INCOMPLETE"
    fi
done
