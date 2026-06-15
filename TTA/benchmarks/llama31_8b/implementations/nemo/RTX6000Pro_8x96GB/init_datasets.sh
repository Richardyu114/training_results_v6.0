#!/bin/bash
# MLPerf Training v6.0 - LLaMA 3.1 8B - Dataset Initialization

set -e

WORKSPACE=${1:-/home/training/mlperf_workspace}

echo "=== Downloading preprocessed C4 dataset ==="
cd "$WORKSPACE"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d llama3_1_8b_preprocessed_c4_dataset \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

echo "=== Downloading LLaMA 3.1 8B tokenizer ==="
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d llama3_1_8b_tokenizer \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

mkdir -p "$WORKSPACE"/{mlperf_output,npy_index,continual_ckpt,nemo_run_logs}

echo "Dataset initialization complete."
