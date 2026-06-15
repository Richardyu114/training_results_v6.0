#!/bin/bash

# Critical fix: increase FD limit (default 1024 caps TCPStore master at ~957 clients)
ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true

# Force libuv TCPStore backend (PyTorch 2.4+) for fast init at scale
export USE_LIBUV=1

# Source the common configs first so per-system overrides win.
for f in /results/config_common.sh /results/config_common_405b.sh /results/config_common_fp4.sh /results/config_common_cg.sh; do
  [ -f "$f" ] && source "$f" 2>/dev/null
done

# DGXSYSTEM-driven per-system config: set by run.sub from each config file's
# `export DGXSYSTEM=$(basename ...)`. Falls back to the proven 1024N baseline
# if DGXSYSTEM is unset or its config file isn't staged in /results.
SYS_CFG="/results/config_${DGXSYSTEM:-GB300_1024x4x8xtp2pp8cp2_cg_fp4}.sh"
if [ -f "$SYS_CFG" ]; then
  source "$SYS_CFG" 2>/dev/null
elif [ -f "/results/config_GB300_1024x4x8xtp2pp8cp2_cg_fp4.sh" ]; then
  source "/results/config_GB300_1024x4x8xtp2pp8cp2_cg_fp4.sh" 2>/dev/null
fi

: "${GPU_ARCH:=gb300}"
: "${EXPERT_PARALLEL:=1}"
export GPU_ARCH EXPERT_PARALLEL

# Inject sitecustomize patch (extends NCCL timeout to 1800s for ckpt load)
export PYTHONPATH="/results/python_patch:${PYTHONPATH:-}"

if [ "${RANK:-0}" -eq 0 ]; then
  echo "WRAPPER_DEBUG: DGXSYSTEM=$DGXSYSTEM SYS_CFG=$SYS_CFG MINIBS=$MINIBS GPU_ARCH=$GPU_ARCH TP=$TENSOR_MODEL_PARALLEL PP=$PIPELINE_MODEL_PARALLEL CP=$CONTEXT_PARALLEL EP=$EXPERT_PARALLEL MBS=$MICRO_BATCH_SIZE IP=$INTERLEAVED_PIPELINE FP4=$FP4 USE_LIBUV=$USE_LIBUV ULIMIT_N=$(ulimit -n)"
fi

./run_and_time.sh "$@" 2>/results/wrapper_stderr_${RANK:-0}.log
EXIT=$?

if [ $EXIT -ne 0 ] && [ "${RANK:-0}" -eq 0 ]; then
  echo "WRAPPER_ERROR: exit code $EXIT"
  echo "WRAPPER_STDERR:"
  tail -30 /results/wrapper_stderr_0.log 2>/dev/null
fi
exit $EXIT
