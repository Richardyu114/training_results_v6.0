# GPT-OSS-20B Pretraining Benchmark

GPT-OSS 20B (Mixture of Experts) using Primus framework.

## Overview

This benchmark trains a 20B parameter GPT model with Mixture of Experts (MoE) architecture using the Primus framework on AMD GPUs.

# 1. Setup Docker Image

## Build Docker Image

Run the following build command from this directory. The build process will take a while to complete.

```bash
# From small_llm_moe_pretraining/primus directory
docker build -t rocm/amd-mlperf:gpt_oss_20b_training_6.0 .
```

For multi-node Slurm/Pyxis runs, also convert the image to an Enroot squashfs (`.sqsh`) on a shared filesystem:

```bash
enroot import dockerd://rocm/amd-mlperf:gpt_oss_20b_training_6.0
# Produces: rocm+amd-mlperf+gpt_oss_20b_training_6.0.sqsh
```

# 2. Prepare Dataset

The current codebase uses the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for training and evaluation.

## Download Preprocessed Data and Model

The pre-tokenized dataset and the tokenizer are available for download. Navigate to your desired download directory and run the following commands:

```bash
# Desired download directory
cd /data/gpt_oss_20b

# Download training and validation data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# Download model tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

mv llama3_1_8b_tokenizer model
```

After download, you should see files with the following naming conventions:
- Training: `c4-train.en_6_text_document.bin` and `.idx`
- Validation: `c4-validation-91205-samples.en_text_document.bin` and `.idx`

The data directory is approximately **80 GB** and model directory is approximately **30 GB**.

# 3. Choose a Launch Path

Two interchangeable launch paths ship with this benchmark; pick whichever fits your environment.

| Launch path                          | Container runtime    | Use when                                                                                  |
|--------------------------------------|----------------------|-------------------------------------------------------------------------------------------|
| `run_with_docker.sh`                 | `docker`             | Single-node bare-metal / Docker Desktop / dev box                                         |
| `job.sh` + `run_with_docker_slurm.sh`| `docker` (per node)  | Slurm cluster *without* Pyxis/Enroot; relies on a working `docker` daemon on every node   |
| `run.sub` + `run_primus_*node.sh`    | Pyxis / Enroot       | Slurm cluster *with* the [Pyxis](https://github.com/NVIDIA/pyxis) plugin (recommended)    |

The training source (`src/train.py`, `conf/*.yaml`, `run_and_time.sh`) is identical across all paths — only the surrounding orchestration changes.

# 4. Run Training (Local Single Node, Docker)

## Set Environment Variables

Set the directory for data and results. Ensure `$LOGDIR` has write access.

```bash
export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results
export CONT=rocm/amd-mlperf:gpt_oss_20b_training_6.0

mkdir -p $LOGDIR
sudo chmod -R 777 $LOGDIR
```

## Set Configuration

Pick the system-specific configuration:

| Config File                                | System | Nodes × GPUs | Global Batch |
|--------------------------------------------|--------|--------------|--------------|
| `config_MI350X_1x8x1_tp1pp1ep1_gbs32.sh`   | MI350X | 1 × 8        | 32           |
| `config_MI350X_2x8x1_tp1pp1ep1_gbs64.sh`   | MI350X | 2 × 8        | 64           |

Source the matching script before launching:

```bash
source config_MI350X_1x8x1_tp1pp1ep1_gbs32.sh
```

## Launch

```bash
export NEXP=1                 # 1 for a single trial, 10 for a submission
bash run_with_docker.sh
```

After completion, logs land under `$LOGDIR`.

# 5. Run Training (Slurm + Pyxis/Enroot, Recommended)

This path uses `srun --container-image=...sqsh` to launch the container on every node — there is no `docker pull` / `docker run` per node, and the same image is reused across trials via `--container-name`.

## Prerequisites

- Slurm cluster with the Pyxis plugin enabled (`srun --container-image=...` works) and Enroot installed (`/usr/bin/enroot`).
- Container image converted to Enroot `.sqsh` on a shared filesystem (Vast/Lustre/NFS) reachable from every compute node — see step 1.
- Shared filesystem visibility for `DATADIR`, `MODELDIR`, and `LOGDIR` from every node.
- InfiniBand / RoCE network with per-GPU RDMA HCAs (`gpu0_rdma` … `gpu7_rdma`) and an Ethernet management interface (`mgmt_eth`). If your cluster uses different interface names (e.g. `ib0`, `eth0`), override `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `UCX_NET_DEVICES` in your config or shell before submitting.

> The host's RDMA libraries (`libibverbs`, `librdmacm`, `libnl-*`, `libionic`, `/etc/libibverbs.d`) are bind-mounted into the container under `/host-rdma/` automatically by `run.sub`, and `LD_LIBRARY_PATH` / `RDMAV_PROVIDERS_PATH` are pointed at them. Mounts are skipped silently for any path that does not exist on the host.

## Set Environment Variables

```bash
# Container image (.sqsh path on the shared filesystem)
export CONT=/path/to/rocm+amd-mlperf+gpt_oss_20b_training_6.0.sqsh

# Data, model, and log directories visible to every node
export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results
mkdir -p "${LOGDIR}" && sudo chmod -R 777 "${LOGDIR}"

# Number of trials (1 for a single run, 10 for submission)
export NEXP=1
```

## Multi-Node via Slurm + Pyxis (2×8)

```bash
export CONFIG_NAME=config_MI350X_2x8x1_tp1pp1ep1_gbs64.sh
export SLURM_PARTITION=<your-partition>       # optional
export SLURM_NODELIST=<node1>,<node2>         # optional pin

bash run_primus_2node.sh
```

Or invoke `sbatch` directly with overrides:

```bash
export CONFIG_NAME=config_MI350X_2x8x1_tp1pp1ep1_gbs64.sh

sbatch \
  --nodes=2 \
  --nodelist=<node1>,<node2> \
  --partition=<your-partition> \
  run.sub
```

## What `run.sub` Does

1. Sources `config_${DGXSYSTEM}.sh` (or `${CONFIG_NAME}` when set) and exports every variable it defines.
2. Resolves `MASTER_ADDR` from the first hostname in `${SLURM_JOB_NODELIST}`.
3. Builds the Pyxis `--container-mounts` list from `DATADIR`, `MODELDIR`, the script directory (mounted at `/workspace/code`), `LOGDIR` (`/results`), the optional `../../utilities` directory (`/workspace/utilities`), and any host RDMA libraries that exist.
4. Builds `--container-env` from every variable defined in the config plus the runtime networking / Slurm vars (`MASTER_ADDR`, `MASTER_PORT`, `SEED`, `SLURM_NNODES`, `SLURM_NODEID`, …).
5. For each of the `NEXP` trials:
   - Optionally drops OS caches on every node (`CLEAR_CACHES=1`, default on).
   - Computes `SEED = SEED_BASE + experiment_index − 1`.
   - Runs `srun -N ${NNODES} --ntasks-per-node=1 --container-image=${CONT} … bash /workspace/code/run_and_time.sh` (one task per node; pyxis launches the container on each).
   - Tees stdout/stderr to `${LOGDIR}/${DATESTAMP}_${i}.log`.
   - Optionally runs `mlperf_logging.compliance_checker` against the trial log when `CHECK_COMPLIANCE=1` (non-blocking).

## Key Networking Variables (set in `run.sub`, overridable in config / shell)

| Variable             | Default                                             | Purpose                                         |
|----------------------|-----------------------------------------------------|-------------------------------------------------|
| `NCCL_SOCKET_IFNAME` | `mgmt_eth`                                          | NCCL bootstrap / control-plane interface         |
| `GLOO_SOCKET_IFNAME` | `mgmt_eth`                                          | Gloo rendezvous interface                        |
| `NCCL_IB_HCA`        | `gpu0_rdma,…,gpu7_rdma`                             | Per-GPU InfiniBand / RoCE HCAs (data plane)      |
| `UCX_NET_DEVICES`    | `gpu0_eth,…,gpu7_eth`                               | UCX transport devices                            |
| `NCCL_DMABUF_ENABLE` | `0`                                                 | Disable DMA-buf (required for ROCm / RCCL)       |
| `SLURM_MPI_TYPE`     | `pmix`                                              | MPI plugin used by `srun`                        |

## Optional Knobs

| Variable                  | Default     | Effect                                                                   |
|---------------------------|-------------|--------------------------------------------------------------------------|
| `NEXP`                    | `1`         | Number of trials (each gets its own SEED and log file)                   |
| `CLEAR_CACHES`            | `1`         | Drop OS page caches on every node before each trial                      |
| `APPLY_RUNTIME_TUNABLES`  | `0`         | Run `runtime_tunables.sh` on every node before training                  |
| `CHECK_COMPLIANCE`        | `0`         | Run `mlperf_logging.compliance_checker` after each trial (non-blocking)  |
| `MLPERF_RULESET`          | `6.0.0`     | Ruleset version passed to the compliance checker                         |
| `WALLTIME_RUNANDTIME`     | `05:00:00`  | `srun --time=` for the training step                                     |
| `CPUS_PER_TASK`           | `128`       | Cores per Slurm task                                                     |
| `MASTER_PORT`             | `29501`     | torch.distributed master port                                            |
| `CONT_NAME`               | `mlperf_primus_gpt_oss_20b` | Logical pyxis container name (reused across trials)      |
| `SEED` / `SEED_BASE`      | `$RANDOM`   | Override the seed used for trial 1                                       |

## Monitoring the Job

```bash
# Slurm status
squeue -u $USER

# Stream the per-trial training log (datestamp printed by run.sub)
tail -f $LOGDIR/<datestamp>_1.log

# Slurm output (stdout/stderr from run.sub itself)
tail -f slurm-<JOBID>.out
```

## Multiple Runs (Submission)

```bash
export NEXP=10
bash run_primus_2node.sh
```

Each run is seeded as `SEED = SEED_BASE + experiment_index − 1` and produces a separate `${DATESTAMP}_${i}.log` under `${LOGDIR}`.

# 6. Run Training (Slurm without Pyxis, Docker per node)

If your cluster does not have Pyxis/Enroot, use the legacy Docker-per-node path (`job.sh` + `run_with_docker_slurm.sh`). Update `job.sh` for your partition / node list / image and submit with `sbatch job.sh`.

# 7. Quality Metrics

## Quality Metric

Validation loss (log perplexity)

## Quality Target

Validation log perplexity = **3.34**

## Evaluation Frequency

Evaluation every **384 iterations** (12,288 samples with GBS=32)

## Evaluation Thoroughness

We evaluate using **1024 sequences** from the validation dataset.

# 8. Directory Structure

```
small_llm_moe_pretraining/primus/
├── conf/                                       # Primus YAML configs (model + training)
│   ├── gpt_oss_20B-pretrain.yaml
│   └── gpt_oss_20B-pretrain-fp8.yaml
├── scripts/
│   └── check_gpus_idle.sh                      # Used by run_with_docker_slurm.sh
├── src/                                        # Training source code (do not modify casually)
│   ├── train.py
│   └── _log_suppression.py
├── config_MI350X_1x8x1_tp1pp1ep1_gbs32.sh      # 1×8 system config
├── config_MI350X_2x8x1_tp1pp1ep1_gbs64.sh      # 2×8 system config
├── Dockerfile
├── build_docker.sh
├── requirements.txt
├── primus_mllog-0.1.21-py3-none-any.whl
├── tune_gemm_results.txt                       # hipBLASLt tuned GEMMs
├── runtime_tunables.sh                         # Optional OS-level perf tunables
├── run_and_time.sh                             # In-container training entry point
│
├── run_with_docker.sh                          # Local single-node Docker launcher
├── job.sh                                      # Legacy: SBATCH → run_with_docker_slurm.sh
├── run_with_docker_slurm.sh                    # Legacy: per-node docker pull / run / exec
│
├── run.sub                                     # Pyxis/Enroot SBATCH script (recommended)
├── run_primus_1node.sh                         # Convenience: sbatch --nodes=1 run.sub
└── run_primus_2node.sh                         # Convenience: sbatch --nodes=2 run.sub
```
