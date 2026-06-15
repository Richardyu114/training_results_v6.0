#!/bin/bash
# MLPerf Training v6.0 - LLaMA 3.1 8B - Run and Time
# System: 8x NVIDIA RTX PRO 6000 Blackwell (96GB)
# Closed Division

set -e

WORKSPACE=${1:-/home/training/mlperf_workspace}
RUN_ID=${2:-1}

echo "============================================"
echo " MLPerf Training v6.0 - LLaMA 3.1 8B"
echo " Closed Division"
echo " 8x RTX PRO 6000 Blackwell (96GB)"
echo " Run #${RUN_ID}"
echo " Started: $(date)"
echo "============================================"

rm -rf "$WORKSPACE/nemo_run_logs"/* 2>/dev/null || true
rm -f "$WORKSPACE/mlperf_output/mlperf_llama31_8b.log" 2>/dev/null || true

docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --network=host \
  --shm-size=256g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cpuset-cpus="0-55,112-167" \
  --cpuset-mems="0" \
  -e PREPROCESSED_PATH="/data/dataset" \
  -e TOKENIZER_PATH="/data/tokenizer" \
  -e GBS=16 \
  -e MBS=1 \
  -e MAX_LR="4e-4" \
  -e WARMUP_STEPS=16 \
  -e EVAL_EVERY=12288 \
  -e START_EVAL_AT=0 \
  -e NCCL_IB_DISABLE=1 \
  -v "${WORKSPACE}/llama3_1_8b_preprocessed_c4_dataset:/data/dataset:ro" \
  -v "${WORKSPACE}/llama3_1_8b_tokenizer:/data/tokenizer:ro" \
  -v "${WORKSPACE}/mlperf_output:/results" \
  -v "${WORKSPACE}/mlperf_output:/mlperf-outputs" \
  -v "${WORKSPACE}/npy_index:/data/npy_index" \
  -v "${WORKSPACE}/continual_ckpt:/data/continual_ckpt" \
  -v "${WORKSPACE}/submission_bs16_fp8/TTA/benchmarks/llama31_8b/implementations/nemo:/workspace/code" \
  -v "${WORKSPACE}/nemo_run_logs:/root/.nemo_run" \
  -w /workspace/code \
  mlperf-nvidia:llama31_8b-pyt \
  bash -c '
    source config_RTX6000Pro_1x8x1_8b.sh
    bash run_llama31.sh
  '

RESULT_DIR="${WORKSPACE}/submission_bs16_fp8/TTA/results/RTX6000Pro_8x96GB/llama31_8b"
mkdir -p "$RESULT_DIR"
cp "$WORKSPACE/mlperf_output/mlperf_llama31_8b.log" "$RESULT_DIR/result_${RUN_ID}.txt"

echo "============================================"
echo " Run #${RUN_ID} Complete: $(date)"
echo " Result saved to: $RESULT_DIR/result_${RUN_ID}.txt"
echo "============================================"
