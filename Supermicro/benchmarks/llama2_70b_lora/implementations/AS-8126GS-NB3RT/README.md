# Supermicro AS-8126GS-NB3RT — MLPerf Training v6.0 LLaMA2-70B LoRA

## System

2x Supermicro AS-8126GS-NB3RT, each with 8x NVIDIA B300 SXM6 (288GB HBM3e), interconnected via NVIDIA Quantum-2 Q3200 NDR InfiniBand switch using 8x NVIDIA ConnectX-8 XDR HCAs per node.

## Software

- **Container**: `nvcr.io/nvdlfwea/mlperftv60/llama2_70b_lora-amd:20260507`
- **Framework**: PyTorch NVIDIA Release 26.04; NeMo Framework NVIDIA Release 26.04
- **CUDA**: 13.2; Driver 595.58.03
- **OS**: Ubuntu 24.04.2 LTS, kernel 6.8.0-110-generic
- **Workload manager**: Slurm 23.11.4 + Pyxis + Enroot

## Configuration

- `DGXNNODES=2`, `DGXNGPU=8`
- `MINIBS=1`, `TP=1`, `PP=1`, `CP=1`
- `LR=0.00055`, `FP8_ACT=1`, `NCCL_NVLS_ENABLE=1`
- See `HGXB300_2x8x1xtp1pp1cp1` config in container

## Steps to reproduce

1. Run `setup.sh` once to log in to NGC and prepare the squashfs container image on both nodes.
2. Run `init_datasets.sh` once to download and preprocess the GovReport dataset and Llama-2-70B base checkpoint.
3. Run `run_and_time.sh` to launch 10 timed runs via sbatch.

## Results

10/10 runs PASS with full MLPerf v6.0 compliance log.
- Median: ~368.7s (6.15 min)
- Min: ~346.7s, Max: ~457.8s
