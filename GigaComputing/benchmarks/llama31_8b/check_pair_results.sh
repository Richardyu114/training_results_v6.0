#!/bin/bash
# ============================================================
# check_pair_results.sh
# 診斷結束後，彙整各 pair 的 throughput / error 狀況
# 用法: bash check_pair_results.sh
# ============================================================

LOGDIR="/home/mlperf_training/llama31_8b/results"
JOBID_LOG="$(dirname "${BASH_SOURCE[0]}")/pair_diag_jobids.txt"

if [ ! -f "$JOBID_LOG" ]; then
    echo "找不到 $JOBID_LOG，請先跑 run_pair_diag.sh"
    exit 1
fi

echo "=========================================="
echo " Pair Diagnostic Result Summary"
echo " $(date)"
echo "=========================================="
printf "%-30s %-10s %-20s %s\n" "PAIR" "JOB_ID" "THROUGHPUT(samples/s)" "STATUS"
echo "----------------------------------------------------------------------------------------------------------"

while read -r JOBID PAIR TAG; do
    OUTFILE=$(ls "${LOGDIR}/diag_${TAG}_${JOBID}.out" 2>/dev/null | head -1)
    ERRFILE=$(ls "${LOGDIR}/diag_${TAG}_${JOBID}.err" 2>/dev/null | head -1)

    if [ -z "$OUTFILE" ] || [ ! -f "$OUTFILE" ]; then
        # job 可能還在跑，查 squeue
        SQSTATE=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)
        if [ -n "$SQSTATE" ]; then
            printf "%-30s %-10s %-20s %s\n" "$PAIR" "$JOBID" "-" "RUNNING ($SQSTATE)"
        else
            printf "%-30s %-10s %-20s %s\n" "$PAIR" "$JOBID" "-" "NOT FOUND / PENDING"
        fi
        continue
    fi

    # 從 log 抓 throughput（MLPerf 通常會印 samples/s 或 seq/s）
    THROUGHPUT=$(grep -oP 'throughput[=:]\s*\K[0-9.]+' "$OUTFILE" 2>/dev/null | tail -1)
    if [ -z "$THROUGHPUT" ]; then
        THROUGHPUT=$(grep -oP '[0-9.]+ samples/s' "$OUTFILE" 2>/dev/null | tail -1)
    fi
    if [ -z "$THROUGHPUT" ]; then
        THROUGHPUT="N/A"
    fi

    # 檢查有沒有 error
    HAS_ERR=""
    if grep -qiE "error|exception|killed|oom|cuda|nccl" "$ERRFILE" 2>/dev/null; then
        HAS_ERR=" ⚠ ERR: $(grep -iEm1 'error|exception|killed|oom|cuda|nccl' "$ERRFILE")"
    fi

    # 判斷 job 是否完成
    SACCT_STATE=$(sacct -j "$JOBID" -n -o State 2>/dev/null | head -1 | tr -d ' ')
    STATUS="${SACCT_STATE:-UNKNOWN}"

    printf "%-30s %-10s %-20s %s%s\n" "$PAIR" "$JOBID" "$THROUGHPUT" "$STATUS" "$HAS_ERR"

done < "$JOBID_LOG"

echo ""
echo "提示: throughput 明顯偏低或有 ERR 的 pair → 該組 node 可能有問題"
echo "      再用 'scontrol show node <nodename>' 或 'nvidia-smi' 進一步確認"
