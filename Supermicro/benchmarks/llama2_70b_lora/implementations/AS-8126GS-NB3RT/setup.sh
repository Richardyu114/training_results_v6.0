#!/usr/bin/env bash
# One-time setup: pull NGC container to both nodes and create local sqsh.
# Run on the head node (node1). Requires SSH access to node2 and NGC credentials.
set -euo pipefail

: "${CONT_URL:=nvcr.io/nvdlfwea/mlperftv60/llama2_70b_lora-amd:20260507}"
: "${SQSH_NAME:=llama2_70b_lora_20260507.sqsh}"
: "${SQSH_DIR:=$HOME/sqsh}"
: "${NODE2:=b300-node2}"

mkdir -p "${SQSH_DIR}"

# Make sure NGC credentials are configured (see ~/.config/enroot/.credentials).
echo "[*] Pulling ${CONT_URL} on $(hostname)..."
enroot import -o "${SQSH_DIR}/${SQSH_NAME}" "docker://${CONT_URL}"

echo "[*] Pulling ${CONT_URL} on ${NODE2}..."
ssh "${NODE2}" "mkdir -p ${SQSH_DIR} && enroot import -o ${SQSH_DIR}/${SQSH_NAME} docker://${CONT_URL}"

echo "[*] Done. sqsh ready at ${SQSH_DIR}/${SQSH_NAME} on both nodes."
