#!/usr/bin/env bash
# One-time dataset and model checkpoint download for LLaMA2-70B LoRA.
# Run inside the container interactively; dataset goes to host-mounted /data.
set -euo pipefail

: "${DATAROOT:=$HOME/training60}"
: "${CONT_URL:=nvcr.io/nvdlfwea/mlperftv60/llama2_70b_lora-amd:20260507}"

mkdir -p "${DATAROOT}/data/gov_report" "${DATAROOT}/model-v2"

echo "[*] Need at least 300GB free at ${DATAROOT}. Current free space:"
df -h "${DATAROOT}"

cat <<EOF

Run this docker command interactively:

  docker run -it --rm --network=host --ipc=host \\
    --volume ${DATAROOT}:/data \\
    ${CONT_URL}

Then inside the container:

  python scripts/download_dataset.py --data_dir /data/data/gov_report
  python scripts/download_model.py --model_dir /data/model-v2

Expected output:
  /data/data/gov_report/{train.npy, validation.npy}
  /data/model-v2/{context/, weights/}

EOF
