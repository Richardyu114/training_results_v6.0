# Supermicro AS-8126GS-NB3RT — MLPerf Training v6.0 LLaMA3.1 8B

## System

2x Supermicro AS-8126GS-NB3RT, each with 8x NVIDIA B300 SXM6 (288GB HBM3e), interconnected via NVIDIA Quantum-2 Q3200 NDR InfiniBand switch using 8x NVIDIA ConnectX-8 XDR HCAs per node.

## Software

- **Container**: `nvcr.io/nvdlfwea/mlperftv60/llama31_8b-amd:20260507`
- **Framework**: PyTorch NVIDIA Release 26.04; NeMo Framework NVIDIA Release 26.04
- **CUDA**: 13.2; Driver 595.58.03
- **OS**: Ubuntu 24.04.2 LTS, kernel 6.8.0-110-generic
- **Workload manager**: Slurm 23.11.4 + Pyxis + Enroot

## Configuration

- `DGXNNODES=2`, `DGXNGPU=8`
- `MINIBS=2`, `TP=1`, `PP=1`, `CP=1`
- `NCCL_SOCKET_FAMILY=AF_INET`, `NCCL_IGNORE_COLLNET_MISMATCH=1`
- `MLPERF_RULESET=6.0.0`
- See `HGXB300_2x8x2xtp1pp1cp1_8b` config in container

## Steps to reproduce

1. Run `setup.sh` once to log in to NGC and prepare the squashfs container image on both nodes.
2. Run `init_datasets.sh` once to download and preprocess the C4-en pretraining dataset.
3. Run `run_and_time.sh` to launch 10 timed runs via sbatch.

## Results

10/10 runs PASS with full MLPerf v6.0 compliance log (val_loss < 3.3).
- Median: ~55.0 min
