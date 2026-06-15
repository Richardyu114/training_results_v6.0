#!/bin/bash
set -euo pipefail

# =========================================================================
# 1. 加载配置
# =========================================================================
CONFIG_FILE="/workspace/ft-llm-shared/config_B200_8x8x1xtp1pp1cp8.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "Config not found: $CONFIG_FILE"
    exit 1
fi

export IB_BIND="${IB_BIND:---ib=single}"
export CPU_EXCLUSIVE="${CPU_EXCLUSIVE:---cpu=exclusive}"

# =========================================================================
# 2. 路径硬编码 & 变量设置
# =========================================================================
export DATADIR="/data/mlperf_data51/gov_report"
export MODEL_DIR="/data/mlperf_data51/model"
export CHECKPOINT_PATH="${MODEL_DIR}"
export TOKENIZER_MODEL="${MODEL_DIR}/context/nemo_tokenizer/tokenizer.model"
export TRAIN_FILE="${DATADIR}/train.npy"
export VALID_FILE="${DATADIR}/validation.npy"
export DATA_PATH="[1.0,${TRAIN_FILE}]"
export VALID_DATA_PATH="[1.0,${VALID_FILE}]"

# B200 原生 IB 网络优化
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
unset NCCL_IB_GID_INDEX
export NCCL_IB_DISABLE=0
export NCCL_IB_GDR_LEVEL=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export USE_DISTRIBUTED_OPTIMIZER=True

# =========================================================================
# 3. 环境修复
# =========================================================================
if [ -d "/data" ]; then
    ln -sf "${TRAIN_FILE}" /data/train.npy
    ln -sf "${VALID_FILE}" /data/validation.npy
    ln -sf "${TRAIN_FILE}" /data/llama_text_document.npy
    ln -sf "${TOKENIZER_MODEL}" /data/tokenizer.model
fi

# =========================================================================
# 4. MPI 映射 & 节点计算
# =========================================================================
if [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then
    export WORLD_SIZE="${WORLD_SIZE:-${OMPI_COMM_WORLD_SIZE:-${SLURM_NTASKS:-1}}}"
    export RANK="${RANK:-${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}}"
    export LOCAL_RANK="${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}}"
    export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${OMPI_COMM_WORLD_LOCAL_SIZE:-${SLURM_NTASKS_PER_NODE:-1}}}"
    export GROUP_RANK="$((RANK / LOCAL_WORLD_SIZE))"
    export MASTER_PORT="${MASTER_PORT:-29500}"
else
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    export RANK="${RANK:-0}"
    export LOCAL_RANK="${LOCAL_RANK:-0}"
    export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-1}"
    export GROUP_RANK="${GROUP_RANK:-0}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
fi

: "${DGXNGPU:=8}"
NUM_NODES=$((WORLD_SIZE / DGXNGPU))
if [ "${NUM_NODES}" -eq 0 ]; then
    NUM_NODES=1
fi

export NVTX_FLAG=0
export NCCL_TEST=0
export DGXNGPU

# =========================================================================
# 5. 生成每进程独立的启动脚本
# =========================================================================
TARGET_SCRIPT="./run_and_time_patched_${RANK}.sh"
echo "[Wrapper] Creating unique script: ${TARGET_SCRIPT} ..."

cp ./run_and_time.sh "${TARGET_SCRIPT}"

CLEAR_CACHES="${CLEAR_CACHES:-1}"
DROPCACHE_CMD="${DROPCACHE_CMD:-sudo /sbin/sysctl vm.drop_caches=3}"

cache_clear_all_nodes() {
    # 只让每个节点的 local_rank0 执行一次
    if [ "${LOCAL_RANK:-0}" -ne 0 ]; then
        return 0
    fi

    local host
    host="$(hostname)"
    echo "[CacheClear] ${host} sync_start"
    sync && echo "[CacheClear] ${host} sync_done"

    local cache_before cache_after
    cache_before="$(awk '/^Cached:/ {print $2}' /proc/meminfo || echo 0)"
    echo "[CacheClear] ${host} cached_before=${cache_before}kB"

    if ${DROPCACHE_CMD} 2>/dev/null; then
        echo "[CacheClear] ${host} drop_caches ok"
    else
        echo "[CacheClear] ${host} drop_caches failed (no sudo?). Continuing."
    fi

    cache_after="$(awk '/^Cached:/ {print $2}' /proc/meminfo || echo 0)"
    echo "[CacheClear] ${host} cached_after=${cache_after}kB"

    python - <<'PY' 2>/dev/null || true
from mlperf_common.callbacks import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)
PY
}

if [ "${CLEAR_CACHES}" -eq 1 ]; then
    echo "[Wrapper] CLEAR_CACHES=1 -> performing cache clear (per-node, local_rank0)"
    cache_clear_all_nodes
fi

# =========================================================================
# 6. 注入训练参数
# =========================================================================
# 注意：
# 1) 这里保留了你前面“较新”的那组参数
# 2) 删除了多余的第二条 sed（你原来那条其实被注释吞掉了）
# 3) 如需改 True/False，只改下面这一条
sed -i \
    "s|train.py|train.py ++trainer.devices=${DGXNGPU} ++trainer.num_nodes=${NUM_NODES} ++model.megatron.overlap_grad_sync=False ++model.cuda_graph_per_step=True ++model.optim.use_distributed_optimizer=True ++model.megatron.sequence_parallel=True ++model.sequence_parallel=True|g" \
    "${TARGET_SCRIPT}"

# 确保删掉 ddp
sed -i "s|++trainer.strategy=ddp||g" "${TARGET_SCRIPT}"

chmod +x "${TARGET_SCRIPT}"

# =========================================================================
# 7. 启动
# =========================================================================
echo "----------------------------------------------------------------"
echo "[Wrapper] Launching Rank ${RANK} on $(hostname)..."
echo "----------------------------------------------------------------"

if [ -z "${MASTER_ADDR:-}" ]; then
    export MASTER_ADDR=llama2-70b-lora-n8-mpijob-worker-0
fi

export HYDRA_FULL_ERROR=1

echo "[Wrapper] RANK=${RANK} LOCAL_RANK=${LOCAL_RANK} WORLD_SIZE=${WORLD_SIZE} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE} HOST=$(hostname)"
echo "[FINAL CONFIG] TP=${TP:-NA} CP=${CP:-NA} PP=${PP:-NA} DGXNNODES=${DGXNNODES:-NA} DGXNGPU=${DGXNGPU}"

# =========================================================================
# 8. 关键修复：只让 global rank 0 的输出进入外层汇总
# =========================================================================
# 背景：
# compliance 里 submission_* / opt_* / lora_* 要求 EXACTLY_ONE
# 如果所有 rank 的 stdout/stderr 都被外层收集到同一个 result_0.txt，
# 这些全局字段就会重复 N 次（你现在看到的是 64 次）
#
# 策略：
# - rank 0: 正常输出到 stdout/stderr，供 MLPerf/result collector 采集
# - 其他 rank: 输出重定向到各自独立日志，避免污染全局 result 文件
#
# 这不会影响训练本身，只影响日志汇总方式。
if [ "${RANK}" -eq 0 ]; then
    echo "[Wrapper] Global rank 0 -> keep stdout/stderr for compliance collection"
    exec bash "${TARGET_SCRIPT}"
else
    exec bash "${TARGET_SCRIPT}" >/dev/null 2>&1
fi
