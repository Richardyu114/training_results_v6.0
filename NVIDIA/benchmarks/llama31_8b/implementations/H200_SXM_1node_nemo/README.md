# Llama 3.1 8B — single node 8×H200, bare-metal Docker (no Slurm)

MLPerf Training v6.0 Llama-3.1-8B pretraining on **one node with 8× H200 (SXM or NVL, 141 GB)**,
launched with plain Docker instead of Slurm/Pyxis. FP8 and BF16 configs are provided.

The training code (`Dockerfile`, `conf/`, `run_and_time.sh`, `callback_*.py`, `embedding_lib/`, …)
is the NVIDIA NeMo / Megatron-Bridge reference. What is specific to this directory:

- `config_H200_1x8x2xtp1pp1cp1_8b_fp8.sh` — FP8 hybrid (main)
- `config_H200_1x8x2xtp1pp1cp1_8b_bf16.sh` — BF16 (comparison / fallback)
- `run_with_docker.sh` — bare-metal single-node launcher (this file's reason for existing)
- `pretrain.py` — one small change vs upstream: an `MLPERF_VERBOSE_LOGS` switch that re-enables
  Megatron-Bridge's native per-iteration throughput/TFLOP log (see "Training throughput" below)

## Provenance / sources

- **Framework code** (`Dockerfile`, `conf/`, `run_and_time.sh`, `callback_*.py`,
  `embedding_lib/`, `config_common*.sh`, `data_scripts/`, etc.) is copied **verbatim** from the
  upstream `NVIDIA/benchmarks/llama31_8b/implementations/nemo/` reference in this repo. Not modified.
  Blackwell-only (FP4 / GB200 / GB300 / B200) and 405b configs were removed as not applicable to H200.
  `pretrain.py` is the upstream file with **one** addition: the `MLPERF_VERBOSE_LOGS` gate described
  under "Training throughput" (default-off path is byte-for-byte the upstream behavior).
- **FP8 config** reproduces the verified Dell single-node 8×H200 MLPerf v6.0 submission
  [`Dell/results/1xXE7740x8H200-NVL-141GB/llama31_8b/`](https://github.com/Richardyu114/training_results_v6.0/tree/main/Dell/results/1xXE7740x8H200-NVL-141GB/llama31_8b)
  (Intel Xeon host — same CPU family as our target node). That submission did not ship its config
  file, so the hyperparameters here (GBS=16, MBS=2, LR=4e-4, warmup=16 steps, eval every 768 steps,
  FP8 `hybrid`/`tensorwise`) were reconstructed from that run's MLLOG + printed config dump,
  cross-checked against the structure of the one committed H200 config in the repo,
  [`Cisco/.../config_C885A_H200_2x8x1xtp1pp1cp1_8b.sh`](https://github.com/Richardyu114/training_results_v6.0/blob/main/Cisco/benchmarks/llama31_8b/implementations/nemo/config_C885A_H200_2x8x1xtp1pp1cp1_8b.sh).
  Dell also has a second 8×H200 result
  ([`1xXE7745...`](https://github.com/Richardyu114/training_results_v6.0/tree/main/Dell/results/1xXE7745x8H200-NVL-141GB/llama31_8b),
  AMD EPYC host) using MBS=1/grad_acc=2; we follow the faster XE7740 layout (MBS=2/grad_acc=1).
  All 10 XE7740 trials hit eval log-ppl ≤ 3.3.
- **BF16 config** is **not yet validated on H200** — Dell shipped only FP8 H200 results. It is
  derived by flipping the NeMo `FP8=False` switch on the same layout; convergence/throughput on
  H200 must still be confirmed by our own run.
- **`run_with_docker.sh`** is our own bare-metal launcher (no Slurm), modeled on the env-forwarding
  pattern from `AMD/benchmarks/llama31_8b/implementations/MI308X_MI325X_CDNA3_primus/run_with_docker.sh`,
  adapted to NVIDIA GPU flags and the NeMo container layout.

## Configuration summary

| | value |
|---|---|
| Hardware | 1 node × 8 GPU (H200, 141 GB) |
| Parallelism | TP=1, PP=1, CP=1 → **DP=8** (8B fits on one GPU) |
| Global batch size | 16 (`MINIBS=2 × 8` GPUs) |
| Micro batch size | 2 (grad-accum = 1) |
| Sequence length | 8192 |
| Peak / end LR | 4e-4 / 4e-5, cosine + linear warmup |
| Warmup | 16 steps (= 256 samples) |
| Eval | every 768 steps, target eval log-ppl ≤ 3.3 |
| CUDA Graph | enabled (`FULL_CUDA_GRAPH=1`, Hopper-native) |
| FP8 | hybrid, tensorwise recipe; attention stays BF16 (`FP8_DPA=False`) |

The **FP8** config reproduces the verified Dell XE7740 8×H200 result (converges to ≤3.3 in
~9984 steps, ~4–4.4 h wall clock). The **BF16** config uses the same layout with FP8 disabled and
is **not yet validated on H200** (see Provenance above).

## 1. Requirements

- Docker with the NVIDIA Container Toolkit (`--gpus all` must work)
- ~1 TB disk for the preprocessed C4 dataset
- 8× H200 on a single node

## 2. Build the container

```bash
cd NVIDIA/benchmarks/llama31_8b/implementations/H200_SXM_1node_nemo
docker build -t mlperf-nvidia:llama31_8b-pyt .
export CONT=mlperf-nvidia:llama31_8b-pyt
```

The `Dockerfile` builds on `nvcr.io/nvidia/pytorch:26.04-py3` and adds cuDNN 9.21.1.3,
NCCL v2.30.4, Megatron-Bridge/Megatron-Core 26.04, the benchmark deps, and the custom
embedding CUDA extension. Transformer Engine defaults to the base-image build (`TE_REVISION=SKIP`),
which supports H200 (Hopper, SM90) — no change needed.

## 3. Prepare the dataset

Llama-3.1-8B is trained **from scratch** (no checkpoint). Only the preprocessed C4 dataset and
tokenizer are needed:

```bash
export DATADIR=/path/to/data
bash data_scripts/download_8b.sh
```

Final layout must be `${DATADIR}/8b/{c4-train*.bin,c4-train*.idx,c4-validation*,tokenizer/}`
(the launcher mounts `${DATADIR}/8b` → `/preproc_data` and `${DATADIR}/8b/tokenizer` →
`/workspace/llm/nemo_tokenizer`).

## 4. Launch training

FP8 (main):

```bash
export CONT=mlperf-nvidia:llama31_8b-pyt
export DATADIR=/path/to/data
export LOGDIR=/path/to/logs
CONFIG_FILE=config_H200_1x8x2xtp1pp1cp1_8b_fp8.sh bash run_with_docker.sh
```

BF16 (comparison):

```bash
CONFIG_FILE=config_H200_1x8x2xtp1pp1cp1_8b_bf16.sh bash run_with_docker.sh
```

Quick smoke test (50 steps, does not reach target — just checks the pipeline comes up):

```bash
MAX_STEPS=50 CONFIG_FILE=config_H200_1x8x2xtp1pp1cp1_8b_fp8.sh bash run_with_docker.sh
```

### How the launcher works

`run_with_docker.sh` starts the image detached, sources the chosen config in a clean subshell to
enumerate its exported variables, and forwards them into the container by name (so the container
reads the real values). It then calls `run_and_time.sh` once with `LOCAL_WORLD_SIZE=1`, which makes
`run_and_time.sh` take its single-node interactive branch and fan out 8 ranks itself via
`torchrun --nproc_per_node=8`. No Slurm, Pyxis, Enroot, or `bindpcie` involved.

Useful overrides: `NEXP` (number of trials, default 1), `SEED`, `MASTER_PORT`,
`CLEAR_CACHES=0` (skip the `sudo sysctl vm.drop_caches` step if you lack sudo).

### Training throughput (TFLOP/s/GPU)

Both configs set `MLPERF_VERBOSE_LOGS=1`, which makes `pretrain.py` re-enable Megatron-Bridge's
**native** per-iteration training log every `LOG_EVERY_N_STEPS` (=32). Each line reports the
built-in throughput metric alongside loss and step time, e.g.:

```
[YYYY-MM-DD HH:MM:SS] iteration    32/ 1200000 | consumed samples: ... | elapsed time per iteration (ms): 1447.0 | throughput per GPU (TFLOP/s/GPU): 582.8 | ...
```

This TFLOP number is computed by Megatron-Bridge itself (`num_floating_point_operations()` /
`log_throughput`), the **same** mechanism the AMD Primus logs use — so H200 and MI308/MI325 numbers
are directly comparable. We do not hand-compute it. The upstream NVIDIA submission suppresses this
log (relies on MLLOG only) by setting `log_interval = max_steps+1` and `skip_train_metrics_log`;
our `pretrain.py` gates that suppression behind `MLPERF_VERBOSE_LOGS` so the default-off path stays
byte-for-byte upstream. Set `MLPERF_VERBOSE_LOGS=0` for a clean MLPerf-style run.

As a cross-check: for the FP8 config on 8×H200 the expected steady-state throughput is
~583 TFLOP/s/GPU (from the Dell XE7740 step time of ~1.447 s;
`421.59 × GBS(16) / step_time / 8 GPUs`).

## 5. Quality

- Metric: validation log-perplexity
- Target: ≤ 3.3
- Eval cadence: every 12 288 sequences (`VAL_CHECK_INTERVAL=768` steps at GBS 16)
