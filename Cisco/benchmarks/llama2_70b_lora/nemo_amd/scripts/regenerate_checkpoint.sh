#!/usr/bin/env bash
# Regenerate the Llama2-70B NeMo (zarr) checkpoint using the current training container
# so the saved sharded state-dict includes TransformerEngine `_extra_state` entries.
#
# Why this exists: the pre-staged checkpoints under
#   /mnt/vast/mlperf_data/llama2_70b_finetuning/model/model_weights/
#   /mnt/vast/mlperf_data/models/Llama2-70b-fused-qkv-mlperf/nemo_model/model_weights/
# were produced by an older TE/Megatron pipeline that did not emit `_extra_state` shard
# directories. The current image (amd_rocm_llama2_70b_lora_20260421_dev2) ships newer
# TE/Megatron that *requires* those shards at load time, so restore_from(...) crashes:
#   FileNotFoundError: .../linear_proj._extra_state/shard_0_80.pt
# We fix the cause by re-running the MLPerf-reference conversion inside the same image
# the trainer uses, which guarantees the on-disk layout matches what the trainer loads.
#
# Output layout (all persistent, under one configurable training root):
#   ${TRAINING_ROOT}/hf_cache/          -> HF_HOME (~140 GB Llama-2-70B-hf snapshot)
#   ${TRAINING_ROOT}/llama2_70b_nemo/   -> regenerated NeMo/zarr checkpoint
#
# Runs the conversion inside a detached tmux session so it survives SSH disconnects.
# The token is never written to any file or baked into any command line — it is
# forwarded into the container only via `docker run -e HF_TOKEN` (value-less form),
# which picks up the value from the caller's environment.
#
# Usage:
#   export HF_TOKEN=...          # must have accepted meta-llama/Llama-2-70b-hf license
#   ./scripts/regenerate_checkpoint.sh
#   tmux attach -t llama2_ckpt_regen   # in another terminal
#
# Tunables (override via env before launch):
#   CONT              docker image name (local)
#   OUTPUT_DIR        host path for regenerated checkpoint
#   HF_CACHE_DIR      host path for HF snapshot cache (persistent)
#   LOG_DIR           host path for log files
#   TMUX_SESSION      tmux session name
#   CONT_NAME         docker container name

set -Eeuo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${HF_TOKEN:?HF_TOKEN must be exported before running. Get one at https://huggingface.co/settings/tokens and accept the Llama-2 license on huggingface.co/meta-llama/Llama-2-70b-hf}"

: "${CONT:=amd_rocm_llama2_70b_lora_20260421_dev2:latest}"
# Everything persistent lives under a single training root on NFS.
: "${TRAINING_ROOT:=/mnt/vast/mlperf_data/models/training}"
: "${OUTPUT_DIR:=${TRAINING_ROOT}/llama2_70b_nemo}"
: "${HF_CACHE_DIR:=${TRAINING_ROOT}/hf_cache}"
: "${LOG_DIR:=${CODE_DIR}/results}"
: "${TMUX_SESSION:=llama2_ckpt_regen}"
: "${CONT_NAME:=llama2_ckpt_regen_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}" "${LOG_DIR}"

# The container runs as root; Vast NFS uses root_squash so root-in-container maps
# to `nobody` on the server. Writable mount targets must therefore be world-writable
# (mode 1777) to accept writes from the containerised process.
chmod 1777 "${OUTPUT_DIR}" "${HF_CACHE_DIR}"

LOG_FILE="${LOG_DIR}/ckpt_regen_$(date +%Y%m%d_%H%M%S).log"

command -v tmux   >/dev/null || { echo "ERROR: tmux is required";   exit 1; }
command -v docker >/dev/null || { echo "ERROR: docker is required"; exit 1; }
[[ -e /dev/kfd ]] || { echo "ERROR: /dev/kfd missing on this host"; exit 1; }
docker image inspect "${CONT}" >/dev/null 2>&1 || {
  echo "ERROR: docker image '${CONT}' not found locally"; exit 1; }

if [[ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" ]]; then
  echo "ERROR: OUTPUT_DIR='${OUTPUT_DIR}' is not empty. Refusing to overwrite."
  echo "       Move it aside or set OUTPUT_DIR to a new path."
  exit 1
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session '${TMUX_SESSION}' already exists."
  echo "       Attach:  tmux attach -t ${TMUX_SESSION}"
  echo "       Or kill: tmux kill-session -t ${TMUX_SESSION}"
  exit 1
fi

# Command executed inside the container. No secrets are embedded; HF_TOKEN is
# forwarded in via `docker run -e HF_TOKEN`, and HF_HOME points to a persistent
# host bind-mount so the ~140 GB base model download survives teardowns.
read -r -d '' INNER_CMD <<'EOS' || true
set -Eeuxo pipefail
export HF_HOME=/hf_cache
export PYTHONUNBUFFERED=1

python - <<'PY'
import transformer_engine, megatron.core, nemo
print("TE:",            transformer_engine.__version__)
print("megatron.core:", megatron.core.__version__)
print("nemo:",          nemo.__version__)
PY

# NeMo 2.3's llm collection imports `nemo_run` at module load, and nemo_run itself
# pulls paramiko/fabric/torchx/inquirerpy/leptonai/toml at __init__ time. The AMD
# image doesn't ship any of those. We install nemo_run *with* its deps: pip's
# default upgrade strategy is `only-if-needed`, so it will not touch the in-tree
# NeMo or Megatron installs (they're not in nemo_run's requirement graph).
if ! python -c 'import nemo_run' 2>/dev/null; then
  pip install --no-build-isolation --disable-pip-version-check \
      'git+https://github.com/NVIDIA-NeMo/Run.git'
fi
python -c 'import nemo_run; print("nemo_run:", nemo_run.__version__ if hasattr(nemo_run, "__version__") else "ok")'
# Cheap sanity check: NeMo and Megatron must still import cleanly and report the
# same versions we saw before the nemo_run install.
python -c 'import nemo, megatron.core; print("post-install nemo:", nemo.__version__, "megatron.core:", megatron.core.__version__)'

cd /workspace/code
python scripts/convert_model.py --output_path=/ckpt_regen/

mv /ckpt_regen/weights/ /ckpt_regen/model_weights/
cd /ckpt_regen/model_weights/
shopt -s nullglob
for d in module.*/; do mv -- "$d" "model.${d#module.}"; done
shopt -u nullglob

cp /ckpt_regen/context/nemo_tokenizer/tokenizer.model /ckpt_regen/tokenizer.model
cp /workspace/code/scripts/model_config.yaml          /ckpt_regen/model_config.yaml

echo "=== Validation: looking for _extra_state shard dirs ==="
if ! ls /ckpt_regen/model_weights/ | grep -q _extra_state; then
  echo "FAIL: no _extra_state dirs produced; regeneration did not fix the cause."
  exit 2
fi
ls /ckpt_regen/model_weights/ | grep _extra_state
echo "--- shard count under linear_proj._extra_state ---"
ls /ckpt_regen/model_weights/model.decoder.layers.self_attention.linear_proj._extra_state/ | wc -l
echo "=== Regenerated checkpoint ready at /ckpt_regen ==="
EOS

# Build the docker command as an array so shell quoting is correct.
DOCKER_ARGS=(
  run --rm --init
  --name "${CONT_NAME}"
  --net=host --uts=host --ipc=host
  --device /dev/kfd --device /dev/dri
  --security-opt=seccomp=unconfined
  -e HF_TOKEN
  -e PYTHONUNBUFFERED=1
  # Forward the host's proxy settings so pip/git/requests use the proxy that
  # actually works on this subnet. Docker's CLI auto-injects a *different*
  # proxy from ~/.docker/config.json that is not reachable from this host;
  # the explicit -e here takes precedence and overrides it.
  -e http_proxy -e https_proxy -e HTTP_PROXY -e HTTPS_PROXY
  -e no_proxy -e NO_PROXY
  -v "${OUTPUT_DIR}:/ckpt_regen"
  -v "${HF_CACHE_DIR}:/hf_cache"
  -v "${CODE_DIR}:/workspace/code:ro"
  "${CONT}" bash -lc "${INNER_CMD}"
)

# Compose the shell command tmux will run. Use `printf %q` to escape the docker argv
# so the inner shell parses it exactly once. HF_TOKEN is inherited from THIS shell's
# env into tmux's spawned shell, then into the docker process (via `-e HF_TOKEN`).
printf -v TMUX_CMD '{ date; echo CONT_NAME=%q; echo OUTPUT_DIR=%q; echo HF_CACHE_DIR=%q; ' \
  "${CONT_NAME}" "${OUTPUT_DIR}" "${HF_CACHE_DIR}"
TMUX_CMD+=$'docker'
for a in "${DOCKER_ARGS[@]}"; do
  TMUX_CMD+=" $(printf '%q' "${a}")"
done
TMUX_CMD+=$'; rc=$?; echo "exit=$rc"; exit $rc; } 2>&1 | tee -a '
TMUX_CMD+=$(printf '%q' "${LOG_FILE}")

echo "Starting tmux session '${TMUX_SESSION}'"
echo "  docker container:   ${CONT_NAME}"
echo "  output:             ${OUTPUT_DIR}"
echo "  HF cache:           ${HF_CACHE_DIR}"
echo "  log file:           ${LOG_FILE}"
echo
echo "Attach:  tmux attach -t ${TMUX_SESSION}"
echo "Tail:    tail -f ${LOG_FILE}"
echo

tmux new-session -d -s "${TMUX_SESSION}" "${TMUX_CMD}"
