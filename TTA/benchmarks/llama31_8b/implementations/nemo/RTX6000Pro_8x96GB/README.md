# MLPerf Training v6.0 - LLaMA 3.1 8B (Closed Division)

## Submitter
TTA

## System (Klimax-4082)

- **Hardware**: 1x Node, 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB GDDR7)
- **CPU**: AMD EPYC 9654 96-Core (2.4GHz) x2
- **RAM**: 1.5TB DDR5
- **Storage**: 3.6TB NVMe (Samsung SSD 9100 PRO)
- **GPU Interconnect**: PCIe Gen5 x16 (PIX topology, all GPUs on NUMA node 0)

## Software

- **Container**: mlperf-nvidia:llama31_8b-pyt (based on nvcr.io/nvidia/pytorch:25.09-py3, NeMo 25.09-alpha.rc1)
- **Framework**: NeMo 2.6.0rc0 + Megatron-LM core_r0.11.0
- **PyTorch**: 2.9.0a0+50eac811a6.nv25.09
- **CUDA**: 13.0 / Driver 595.58.03
- **NCCL**: 2.27.7
- **Transformer Engine**: 2.8.0

## Configuration

| Parameter | Value |
|---|---|
| Global Batch Size | 32 |
| Micro Batch Size | 1 |
| Gradient Accumulation Steps | 4 |
| Tensor Parallel | 1 |
| Data Parallel | 8 |
| Learning Rate | 8.5e-4 |
| End Learning Rate | 8.5e-5 |
| Warmup Steps | 200 |
| Precision | BF16-mixed + FP8 hybrid |
| Sequence Length | 8192 |
| Target | val_loss <= 3.3 |

## Setup and Execution

```bash
# 1. One-time setup
bash setup.sh

# 2. Download datasets
bash init_datasets.sh /home/training/mlperf_workspace

# 3. Single run
bash run_and_time.sh /home/training/mlperf_workspace 1

# 4. Full submission (10 runs)
bash run_all.sh /home/training/mlperf_workspace
```

## Key Optimizations (Closed Division compliant)

1. FP8 hybrid precision (E4M3 fwd / E5M2 bwd) via Transformer Engine
2. NCCL Ring algorithm for PCIe PIX topology
3. NUMA node 0 CPU affinity binding (--cpuset-cpus, --cpuset-mems)
4. NVTE LayerNorm SM margin for communication-compute overlap
5. TORCH_NCCL_HIGH_PRIORITY for reduced communication latency
6. Distributed optimizer (optimizer state sharding across DP ranks)
7. Persistent data workers with optimized num_workers=12
8. Learning rate 8.5e-4 with warmup=200 (empirically validated for convergence)

## Hyperparameter Tuning Rationale

Learning rate and warmup steps are classified as "unconstrained" in the MLPerf Training Closed Division rules. Our chosen values (LR=8.5e-4, warmup=200) were selected through systematic experimentation demonstrating consistent convergence:

| Configuration | TTT (hours) | Converged (val_loss ≤ 3.3) |
|---|---|---|
| LR=5e-4, warmup=512 (baseline) | ~8.68 | Yes |
| LR=8e-4, warmup=256 | ~6.51 | Yes |
| LR=8.5e-4, warmup=200 | ~5.88 | Yes |

All configurations converge reliably to the target validation loss. The selected configuration achieves the fastest TTT while maintaining training stability across multiple independent runs.

## RCP Bypass Justification

This submission uses `rcp-bypass` because our optimized hyperparameters (LR=8.5e-4, warmup=200) yield a mean convergence point that slightly exceeds the RCP maximum speedup threshold.

### RCP Comparison

| Metric | RCP Reference (BS=32) | TTA Submission |
|---|---|---|
| Mean convergence samples | 178,859 | 173,229 |
| Speedup | - | 1.0325x |
| RCP Max Speedup allowed | - | 1.0300x |
| Excess | - | 0.0025x |

### 10-Run Convergence Statistics

| Metric | Value |
|---|---|
| Convergence rate | 10/10 (100%) |
| Mean TTT | 355.80 min (5.93 hrs) |
| Std TTT | 14.51 min |
| Min TTT | 327.84 min (Run 5) |
| Max TTT | 378.94 min (Run 1) |
| Mean val_loss | 3.2893 |
| Convergence distribution | 159,712 (1x), 172,000 (8x), 184,288 (2x) samples |

### Per-Run Results

| Run | TTT (min) | val_loss | Convergence Samples |
|---|---|---|---|
| 1 | 378.94 | 3.2791 | 184,288 |
| 2 | 353.29 | 3.2936 | 172,000 |
| 3 | 378.83 | 3.2853 | 184,288 |
| 4 | 353.19 | 3.2941 | 172,000 |
| 5 | 327.84 | 3.2897 | 159,712 |
| 6 | 353.34 | 3.2953 | 172,000 |
| 7 | 352.72 | 3.2789 | 172,000 |
| 8 | 353.33 | 3.2965 | 172,000 |
| 9 | 353.34 | 3.2903 | 172,000 |
| 10 | 353.18 | 3.2900 | 172,000 |

All 10 runs were executed sequentially with identical configurations. No runs were excluded or cherry-picked. The marginal RCP exceedance (0.25%) reflects a well-optimized learning rate configuration that consistently converges with low variance (std=14.51 min).
