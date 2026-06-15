# Running AMD LLama2-70B LoRA PyTorch MLPerf Benchmark

This benchmark performs LLama2-70B LoRA finetuning on the
[GovReport](https://gov-report-data.github.io/) dataset on AMD Instinct MI350X
(`gfx950`) GPUs using NeMo + Megatron-LM.

## Repository Layout

All paths below are relative to this directory
(`Cisco_1/benchmarks/llama2_70b_lora/nemo_amd/`).

| File / Directory                | Purpose                                                                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `Dockerfile`                    | Builds the ROCm 7.0.2 + PyTorch 2.9.1 image used for training (RCCL with InfiniBand, Megatron-LM, NeMo v2.3.0, TransformerEngine). |
| `requirements.txt`              | Pinned Python dependencies installed into the image.                                                                          |
| `config_MI350X_1x8x1.sh`        | Hyperparameters for a single-node, 8-GPU MI350X run.                                                                          |
| `config_MI350X_2x8x1.sh`        | Hyperparameters for a 2-node, 8-GPU/node MI350X run.                                                                          |
| `run_env.sh`                    | Shared environment defaults: paths (`DATADIR`/`MODELDIR`/`LOGDIR`/`CONT`), RCCL/NCCL/UCX/OMPI, optional `CONFIG_FILE` sourcing, and `run.sub` defaults. |
| `run_1node.sh`                  | Convenience wrapper for a single-node Docker run: sources `config_MI350X_1x8x1.sh`, expects you to fill in `DATADIR`/`LOGDIR`/`CONT`, then invokes `run_with_docker.sh`. |
| `run_2node.sh`                  | Convenience wrapper for a 2-node Slurm submission: sources `run_env.sh` (which loads `CONFIG_FILE`, default `config_MI350X_2x8x1.sh`), then `sbatch --export=ALL run.sub`. |
| `run_with_docker.sh`            | Single-node Docker driver. Sources `run_env.sh` with `RUN_ENV_SKIP_CONFIG=1`, requires `DGXSYSTEM`/`CONT`/`DATADIR`, launches the container, runs `runtime_tunables.sh`, then loops `run_and_time.sh` for `NEXP` trials. |
| `run.sub`                       | Pyxis/Slurm batch script invoked by `run_2node.sh`. Requires `CODE_DIR` (set by `run_env.sh`) because Slurm copies the script to spool. |
| `run_and_time.sh`               | Per-rank entry point that exports `RANK`/`LOCAL_RANK`/`WORLD_SIZE` from Slurm vars and launches `python3 src/train.py`.        |
| `runtime_tunables.sh`           | Host-level performance tunables (drop caches, CPU governor, THP, disable NMI watchdog/NUMA balancing/ASLR). Runs before each training trial. |
| `scripts/`                      | Dataset and model download/preprocess utilities and the `prepare_data_and_model.sh` driver. |
| `src/`                          | Training code: `train.py`, callbacks (`custom_callbacks.py`, `memory_callbacks.py`), NeMo PEFT config, and `prof_handler.py`. |

# 1. Setup Docker Image

## Option 1: Pull Docker Image

```bash
docker pull rocm/amd-mlperf:llama2_70b_training_5.1
```

## Option 2: Build Docker Image

Build the image from this directory (`Cisco_1/benchmarks/llama2_70b_lora/nemo_amd/`).
The build is long: it rebuilds RCCL from source with InfiniBand support and builds
TransformerEngine from source for `gfx950`. Ensure scripts in this directory have
the expected permissions (read/execute as appropriate) before building.

```bash
cd Cisco_1/benchmarks/llama2_70b_lora/nemo_amd
docker build -t rocm/amd-mlperf:llama2_70b_training_5.1 .
```

The default target architecture is MI350X (`gfx950`). To build for a different
AMD GPU, override `GPU_ARCH` (and, if needed, `BASE_IMAGE`):

```bash
docker build \
    --build-arg GPU_ARCH=gfx942 \
    -t rocm/amd-mlperf:llama2_70b_training_5.1 .
```

# 2. Prepare Dataset

## General Information

GovReport is a long-document summarization dataset of reports written by U.S.
government research agencies. The version used here is already tokenized and
packed so that each sequence has length 8192.

The model is LLama2-70B with fused QKV. You will need ~270 GB of free disk
space to download and convert the model.

## Download and Preprocess Data & Model

Downloading the model from Hugging Face requires you to:

1. Accept the LLAMA 2 COMMUNITY LICENSE AGREEMENT on Hugging Face.
2. Obtain a Hugging Face access token (`HF_TOKEN`).

Start the container and mount the host directory you want to use as the data
root under `/data` inside the container. In this example we use
`/data/mlperf_llama2`:

```bash
docker run -it -v /data/mlperf_llama2:/data \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    rocm/amd-mlperf:llama2_70b_training_5.1
```

Inside the container, run the download + preprocess driver:

```bash
export HF_TOKEN=<your token>
bash ./scripts/prepare_data_and_model.sh
```

## Verify Data

Data and model files are written under `/data` in the container. After
preprocessing you should see:

- `/data/model/` containing the tokenizer, the converted `.nemo` checkpoint,
  `model_config.yaml`, and the sharded `model_weights/` directory.
- `/data/data/` containing `train.npy` and `validation.npy`.

## Exit Container

```bash
exit
```

# 3. Run Training

### Set Environment

Set the directories for data, model, and results. Make sure `$LOGDIR` exists
and is writable by the user running the benchmark (e.g. create it with `mkdir
-p $LOGDIR` and grant write permission). In this example we use
`/data/mlperf_llama2/results`.

```bash
export DATADIR=/data/mlperf_llama2
export LOGDIR=/data/mlperf_llama2/results
export CONT=rocm/amd-mlperf:llama2_70b_training_5.1
```

How the run scripts use these variables:

- **Slurm (2-node)**: `run_2node.sh` sources `run_env.sh` once. That sets
  paths and cluster-wide RCCL/NCCL/UCX/OMPI variables, defaults
  `CONFIG_FILE=config_MI350X_2x8x1.sh`, sources that config, and sets
  `run.sub` defaults (`NEXP`, `WORK_DIR`, `NCCL_TEST`, `CLEAR_CACHES`,
  `LOG_FREQ`, `MASTER_PORT`, `WALLTIME_RUNANDTIME`, `SEED_BASE`, ...). Then
  `run_2node.sh` calls `sbatch --export=ALL run.sub`. `run.sub` is only the
  Pyxis/Slurm driver and is **not** intended to be invoked directly — Slurm
  copies it to spool, so `$0` no longer resolves to the repo path; this is
  why `run_env.sh` exports `CODE_DIR`.
- **Docker (1-node)**: `run_1node.sh` sources `config_MI350X_1x8x1.sh` and
  invokes `run_with_docker.sh`. `run_with_docker.sh` sets
  `RUN_ENV_SKIP_CONFIG=1` and sources `run_env.sh` so you get the same paths
  and cluster exports **without** reloading a `config_*.sh` (the config is
  already in scope). It then launches the container, runs
  `runtime_tunables.sh`, and loops `run_and_time.sh` for `NEXP` trials.

You can override any of `DATADIR`, `LOGDIR`, `CONT`, `MODELDIR`,
`MLPERF_MODELS_ROOT`, or `CONFIG_FILE` in your shell before invoking
`run_1node.sh` / `run_2node.sh`, or edit `run_env.sh` directly.

### Set Configuration

System-specific hyperparameters live in `config_<system>_<nnodes>x<ngpu>x<minibs>.sh`.
Submission configurations included here:

- `config_MI350X_1x8x1.sh` — 1 node × 8 GPUs × MINIBS 1 (MI350X)
- `config_MI350X_2x8x1.sh` — 2 nodes × 8 GPUs × MINIBS 1 (MI350X)

For a 1-node Docker run, source the 1-node config before launching:

```bash
source config_MI350X_1x8x1.sh
```

For the 2-node Slurm run, `run_2node.sh` selects the 2-node config
automatically via `CONFIG_FILE` (default `config_MI350X_2x8x1.sh`).

### Launch a 1-node Training Run (Docker)

Fill in `DATADIR`, `LOGDIR`, and `CONT` at the top of `run_1node.sh` (or
export them in your shell), then:

```bash
export NEXP=1
bash run_1node.sh
```

For 10 consecutive trials (a full submission run):

```bash
export NEXP=10
bash run_1node.sh
```

You can also invoke `run_with_docker.sh` directly if you have already sourced
the appropriate `config_*.sh` and exported `DATADIR`/`LOGDIR`/`CONT`.

### Launch a 2-node Training Run (Slurm + Pyxis/enroot)

Export `DATADIR`, `LOGDIR`, `CONT`, `MODELDIR`, and (if used)
`MLPERF_MODELS_ROOT`, then:

```bash
export NEXP=1                     # or 10 for a full submission
bash run_2node.sh
```

Do not invoke `run.sub` directly unless you export the same set of variables
that `sbatch --export=ALL` would (in particular `CODE_DIR`, the absolute path
to this directory).

After the runs complete, logs are available under `$LOGDIR` as
`result_<index>.txt`.

**Note**: Before each training trial, `run_with_docker.sh` executes
`runtime_tunables.sh` on the host to apply machine-level performance
settings (drop caches, set CPU governor to `performance`, enable transparent
huge pages, etc.).

# 4. Check Quality

### Quality Metric

Cross entropy loss.

### Quality Target

0.925
