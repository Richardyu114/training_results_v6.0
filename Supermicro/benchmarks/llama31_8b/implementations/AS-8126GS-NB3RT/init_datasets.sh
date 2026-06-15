#!/usr/bin/env bash
# One-time dataset preprocessing for LLaMA3.1 8B pretraining (C4-en).
# Refer to NVIDIA MLPerf v6.0 reference instructions in the container.
set -euo pipefail

: "${DATAROOT:=$HOME/training60}"
: "${CONT_URL:=nvcr.io/nvdlfwea/mlperftv60/llama31_8b-amd:20260507}"

mkdir -p "${DATAROOT}/data/8b"

echo "[*] Run the container interactively and follow the in-container README for C4-en preparation:"
echo
echo "  docker run -it --rm --network=host --ipc=host \\"
echo "    --volume ${DATAROOT}/data:/data \\"
echo "    ${CONT_URL}"
echo
echo "Expected output:"
echo "  ${DATAROOT}/data/8b/{tokenizer/, <preprocessed-shards>}"
