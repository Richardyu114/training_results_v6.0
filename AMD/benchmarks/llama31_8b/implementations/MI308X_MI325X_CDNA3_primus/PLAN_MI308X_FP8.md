# Plan: CDNA3 (MI308X/MI325X) 上跑通 Llama3.1-8B (FP8)

本目录派生自官方 `MI350X_EPYC_9575F_primus/`（MLPerf v6.0，CDNA4/gfx950，FP4/mxfp4），
适配到 **MI308X / MI325X（CDNA3, gfx942）**，把精度从 FP4 换成 **FP8 hybrid**。
仅供内部 enablement / bring-up，**不是合法的 MLPerf closed 提交**。

## 为什么不能直接复用官方 MI350X 目录

MI350X/MI355X 用 FP4 训练，代码证据：

| # | 证据 | 位置（原 MI350X 目录） |
|---|---|---|
| 1 | `fp4: true` / `fp4_recipe: mxfp4` | `conf/llama3.1_8B-pretrain-fp4.yaml:102-103` |
| 2 | `FP4=true` / `FP4_RECIPE=mxfp4` | `config_MI350X_1x8x1.sh:85-86` |
| 3 | `MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR='mxfp4'`（MLPerf closed 官方声明） | `config_MI350X_1x8x1.sh:83` |
| 4 | FP4 GEMM 内核 `_ZN5aiter42f4gemm_bf16_per1x32Fp4_...`（A4W4, MXFP4 1×32） | `a4w4_tuned_gemms.csv` |
| 5 | `NVTE_MXFP4_USE_HADAMARD=1` | `config_MI350X_1x8x1.sh:43` |

(4) 的 aiter FP4 GEMM 依赖 CDNA4 原生 FP4 指令，gfx942 硬件不支持 → 308/325 无法复现 FP4。

## 308 vs 325

同架构 gfx942、同支持 FP8、同不支持 FP4。差异仅显存（308≈192GB、325=256GB）与算力。
→ 代码需求一致，故合并为单目录，仅 config label（可含 LR）不同。

## 本目录相对官方 MI350X 目录的改动

- 新增 `conf/llama3.1_8B-pretrain-fp8.yaml`：精度段 `fp4/mxfp4` → `fp8: true` / `fp8_recipe: hybrid`
- 新增 `config_MI308X_1x8x1.sh` / `config_MI325X_1x8x1.sh`：关 FP4、指向 fp8 yaml、label 改、
  `MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR='fp8'`、`FP8=true`、`PRIMUS_TRAIN_ITERS=50`（冒烟）
- 删除 FP4 专属：`conf/llama3.1_8B-pretrain-fp4.yaml`、`a4w4_tuned_gemms.csv`、`config_MI350X_1x8x1.sh`
- `Dockerfile`（已含 gfx942 编译目标）、`run_*.sh`、`src/`、`precompile_aiter.py`、
  `primus_mllog-*.whl`、`requirements.txt` 原样复用

## 执行前提

- **须在宿主机执行**（当前开发容器无 torch/Primus 且无 docker daemon）。
- 宿主机需 docker + rocm 驱动 + 8 卡访问（`/dev/dri`, `/dev/kfd`）。

## 运行

```bash
cd .../implementations/MI308X_MI325X_CDNA3_primus/
docker build -t rocm/amd-mlperf:llama31_8b_training_6.0 .
export DATADIR=/data/mlperf_llama31_8b/data
export MODELDIR=/data/mlperf_llama31_8b/model/
export LOGDIR=/data/mlperf_llama31_8b/results   # sudo chmod -R 777 $LOGDIR
export CONT=rocm/amd-mlperf:llama31_8b_training_6.0
source config_MI308X_1x8x1.sh   # 或 config_MI325X_1x8x1.sh
export NEXP=1
bash run_with_docker.sh
```

## 风险

- **FP8 yaml 键名（最大不确定点）**：`fp8:` / `fp8_recipe:` 的确切写法需进镜像
  `/workspace/Primus` 的 pre_trainer 配置 schema 确认；若报 unknown config key 据此调整。
- **收敛不保证**：LR 沿用 8e-4（为 FP4 调），FP8 下可能需调；短程 run 先看 loss 下降。
- **不可用于 MLPerf closed 提交**（精度已改）。

## 验证

1. `docker images | grep amd-mlperf` 镜像在。
2. 短程 run（ITERS=50）：log 出现 `STARTING TIMING RUN` → torchrun 8 进程 → loss 下降
   → `RESULT,LLAMA3.1_8B,...` → exit 0。
3. 无 OOM / 无 unknown config key / 无 gfx950-only kernel 报错。
4. 通过后放大 `PRIMUS_TRAIN_ITERS`，按 eval 看 perplexity（target log perplexity 3.3）。
