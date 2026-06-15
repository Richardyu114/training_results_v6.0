# LLama3.1 8B Pretraining Benchmark

LLama3.1 8B pretraining using the Primus framework on AMD MI355X GPUs.

**Key Features:**
- 8B parameter LLama3.1 dense model
- FP8 hybrid precision training
- Megatron backend via Primus

# 1. Setup Docker Image

## Build Docker Image

Run the following build command from this directory:

```bash
docker build -t rocm/amd-mlperf:llama3.1_8b_training_6.0 .
```

# 2. Prepare Dataset

The current codebase uses the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for training and evaluation.

## Download Preprocessed Data and Model

```bash
cd /data/mlperf_llama31_8b

# Download training and validation data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# Download model tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

After download, you should see files with the following naming conventions:
- Training: `c4-train.en_6_text_document.bin` and `.idx`
- Validation: `c4-validation-91205-samples.en_text_document.bin` and `.idx`

The data directory is approximately **80 GB** and model directory is approximately **30 GB**.

# 3. Run Training

## Set Environment Variables

```bash
export DATADIR=/data/mlperf_llama31_8b/data
export MODELDIR=/data/mlperf_llama31_8b/model
export LOGDIR=/data/mlperf_llama31_8b/results
export CONT=rocm/amd-mlperf:llama3.1_8b_training_6.0

mkdir -p $LOGDIR
sudo chmod -R 777 $LOGDIR
```

## Set Configuration
Note: Use config_MI355X_* if your platform is MI355X, and config_MI350_* if your platform is MI350X.

```bash
source config_MI355X_1x8x1.sh
```

## Launch Training

Note: The runtime_tunables.sh script is being executed as part of run_with_docker.sh.

### Single Run

```bash
export NEXP=1
export CONFIG_NAME=config_MI355X_1x8x1.sh
bash run_with_docker.sh
```

### Multiple Runs (for submission)

```bash
export NEXP=10
bash run_with_docker.sh
```

After completion, logs will be available under `$LOGDIR`.

# 4. Running with Slurm (Multi-Node)

This section covers launching training on a Slurm cluster using the Pyxis/Enroot container runtime. The provided `run.sub` batch script and `run_primus_2node.sh` launcher handle multi-node orchestration via `srun` with RDMA-backed inter-node communication.

## Prerequisites

- Slurm cluster with the [Pyxis](https://github.com/NVIDIA/pyxis) plugin (provides `--container-image`, `--container-mounts`, etc. for `srun`)
- Container image in `.sqsh` format (Enroot squashfs), e.g. built from the Dockerfile and converted:
  ```bash
  enroot import dockerd://rocm/amd-mlperf:llama3.1_8b_training_6.0
  # Produces: rocm+amd-mlperf+llama3.1_8b_training_6.0.sqsh
  ```
- Shared filesystem (e.g. Vast, Lustre) accessible from all nodes for `DATADIR`, `MODELDIR`, and `LOGDIR`
- InfiniBand/RoCE network with per-GPU RDMA interfaces (`gpu0_rdma` … `gpu7_rdma`) and Ethernet management interface (`mgmt_eth`) — update `run.sub` if your interface names differ

## Configuration Files

| Config | Nodes | GPUs/Node | Total GPUs |
|--------|-------|-----------|------------|
| `config_MI350X_1x8x1.sh` | 1 | 8 | 8 |
| `config_MI350X_2x8x1.sh` | 2 | 8 | 16 |

Both configs expose the same environment variables; the key difference for multi-node is `NNODES=2`.

## Set Environment Variables

```bash
# Container image (.sqsh path on shared filesystem)
export CONT=<path/to/container.sqsh>

# Data and model directories (must be accessible on all nodes)
export DATADIR=<path/to/dataset>
export MODELDIR=<path/to/tokenizer>

# Results directory (will be created automatically)
export LOGDIR=<path/to/results>

# Number of experiment repetitions (1 for a single run, 10 for submission)
export NEXP=1
```

## Single-Node via Slurm (1x8)

```bash
export CONFIG_NAME=config_MI350X_1x8x1.sh

sbatch \
  --nodes=1 \
  --nodelist=<node> \
  --partition=<partition> \
  run.sub
```

## Multi-Node via Slurm (2x8)

Use the provided launcher script, which sets defaults and calls `sbatch`:

```bash
export CONFIG_NAME=config_MI350X_2x8x1.sh

bash run_primus_2node.sh
```

Or invoke `sbatch` directly, overriding the node list and partition as needed:

```bash
export CONFIG_NAME=config_MI350X_2x8x1.sh

sbatch \
  --nodes=2 \
  --nodelist=<node1,node2> \
  --partition=<partition> \
  run.sub
```

The batch script (`run.sub`) automatically:
1. Resolves `MASTER_ADDR` from the first hostname in `SLURM_JOB_NODELIST`
2. Mounts RDMA libraries from the host into the container under `/host-rdma/`
3. Propagates all config and networking environment variables into each container via `--container-env`
4. Runs one `srun` task per node; each task sets `NODE_RANK=${SLURM_NODEID}` before launching `run_and_time.sh`

## Key Networking Variables (set in `run.sub`)

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_SOCKET_IFNAME` | `mgmt_eth` | NCCL bootstrap / control-plane interface |
| `GLOO_SOCKET_IFNAME` | `mgmt_eth` | Gloo rendezvous interface |
| `NCCL_IB_HCA` | `gpu0_rdma,...,gpu7_rdma` | Per-GPU InfiniBand/RoCE HCAs for data-plane |
| `UCX_NET_DEVICES` | `gpu0_eth,...,gpu7_eth` | UCX transport devices |
| `NCCL_DMABUF_ENABLE` | `0` | Disable DMA-buf (required for ROCm/RCCL) |
| `SLURM_MPI_TYPE` | `pmix` | MPI plugin for `srun` |

> **Note:** Update interface names in `run.sub` if your cluster uses different naming conventions (e.g. `ib0`, `eth0`).

## Optional: Apply Runtime Tunables

To apply OS-level performance tunables on all nodes before training (drops caches, sets CPU governor to performance, disables NUMA balancing, enables transparent huge pages):

```bash
export APPLY_RUNTIME_TUNABLES=1
```

This triggers `runtime_tunables.sh` to run via `srun` across all nodes before the training loop begins.

## Monitoring the Job

```bash
# Check job status
squeue -u $USER

# Stream logs in real time (log file is named <DATESTAMP>_<EXP_INDEX>.log)
tail -f $LOGDIR/<datestamp>_1.log

# Slurm output (stdout/stderr from run.sub itself)
tail -f slurm-<JOBID>.out
```

## Multiple Runs (Submission)

```bash
export NEXP=10
bash run_primus_2node.sh
```

Each run is seeded differently (`SEED = SEED_BASE + experiment_index - 1`) and produces a separate log file in `$LOGDIR`.

# 5. Quality Metrics

## Quality Metric

Validation loss (log perplexity)

## Quality Target

Validation log perplexity = **3.3**

## Evaluation Frequency

Evaluation every **384 iterations** (12,288 samples with GBS=32)

## Evaluation Thoroughness

We evaluate using **1024 sequences** from the validation dataset.

# 6. Model Architecture

| Parameter | Value |
|-----------|-------|
| Model Size | 8B parameters |
| Architecture | LLama3.1 (dense) |
| Sequence Length | 8192 |
| Precision | BF16 / FP8 hybrid |
| Hidden Size | 4096 |
| Layers | 32 |
| Attention Heads | 32 |

# 7. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Micro Batch Size | 2 |
| Global Batch Size | 32 |
| Learning Rate | 8e-4 |
| LR Schedule | Cosine decay with warmup |
| Weight Decay | 0.1 |
| Adam b1, b2 | 0.9, 0.95 |
| Training Iterations | 1,200,000 |

# 8. Directory Structure

```
small_llm_pretraining/primus/
├── conf/                               # Configuration files
│   └── llama3.1_8B-pretrain-fp8.yaml
├── dev/                                # Development scripts
│   ├── Dockerfile
│   ├── build_docker.sh
│   ├── run_docker.sh
│   ├── run_and_time.sh
│   └── run_with_docker_dev.sh
├── src/                                # Training source code
│   └── train.py
├── config_MI350X_1x8x1.sh             # Single-node (1x8) configuration
├── config_MI350X_2x8x1.sh             # Multi-node (2x8) configuration
├── Dockerfile
├── run_and_time.sh                     # Training entry point (runs inside container)
├── run_with_docker.sh                  # Single-node Docker launcher
├── run_primus_2node.sh                 # Multi-node Slurm launcher (calls sbatch)
├── run.sub                             # Slurm batch script (sbatch entry point)
├── runtime_tunables.sh                 # OS-level performance tuning (optional)
├── requirements.txt
└── primus_mllog-0.1.1-py3-none-any.whl
```
