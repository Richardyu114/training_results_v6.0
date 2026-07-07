# Running AMD LLama3.1-8B Pretraining PyTorch MLPerf Benchmark (Primus) — CDNA3 (MI308X/MI325X)

Small Language Model pretraining - Llama 3.1 8B using the Primus framework.

> **CDNA3 variant.** This directory is derived from the official MLPerf v6.0 MI350X/MI355X
> (CDNA4, gfx950) submission, adapted for **MI308X / MI325X (CDNA3, gfx942)**. CDNA3 has no
> native FP4, so training precision is switched from **FP4/mxfp4** to **FP8 hybrid**
> (`conf/llama3.1_8B-pretrain-fp8.yaml`). This is for **internal enablement / bring-up only**
> and is **not a valid MLPerf closed submission** (precision differs from the ruleset).
> See section 5 below for the full rationale and code evidence.

# 1. Setup Docker Image

##  Build Docker Image

Run the following build command from the root of the repository. The build process will take a while to complete. Ensure that all scripts have write access by running `sudo chmod -R 777`.

```bash
docker build -t rocm/amd-mlperf:llama31_8b_training_6.0 .
```

**Speeding up the build (CDNA3 only).** The default builds kernels for both `gfx950;gfx942`, which
is slow (the TransformerEngine / AITER CK-kernel compile is the long pole). Since MI308X and MI325X
are both `gfx942`, you can build for CDNA3 alone via the `GPU_ARCHS` build-arg to roughly halve
build time:

```bash
docker build --build-arg GPU_ARCHS="gfx942" -t rocm/amd-mlperf:llama31_8b_training_6.0 .
```

The resulting image runs on both MI308X and MI325X (it just won't run on gfx950/CDNA4).
# 2. Prepare Dataset and Model

The current codebase is using the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for train and evaluation.

> **Download on the host, not inside the container.** The Docker image does **not** contain the
> dataset or model (the Dockerfile only clones code). You download them once on the host, then
> `run_with_docker.sh` bind-mounts them into the container (`DATADIR -> /data`, `MODELDIR -> /model`).
> The same host copy is shared by both the MI308X and MI325X runs — no need to download twice.

## Download Preprocessed Data

The pre-tokenized dataset and the tokenizer are available for download. Pick **any** host directory
you like — you are not tied to `/data`. Set `BASE_DIR` to your chosen location and run the commands
below. (Everything downstream keys off `DATADIR` / `MODELDIR` env vars, so the on-host path is
entirely up to you; only the in-container paths `/data` and `/model` are fixed.)

```bash
# Choose your own host directory (example: a fast NVMe mount)
export BASE_DIR=/mnt/nvme/mlperf_llama31_8b     # <-- change to whatever you want
mkdir -p "${BASE_DIR}"
cd "${BASE_DIR}"

# data (~80 GB) -> creates ./data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# model / tokenizer (~30 GB) -> downloads directly into ./model
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

After the download is complete, you should see files with the following naming conventions under the data directory, ending with both `.idx` and `.bin`: 
- Training partitions: `c4-train.en_6_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

The data directory is ~80 GB and model directory is ~30 GB (a full HF repo is larger, ~45 GB, and is also fine).

# 3. Run Training

### Set Environment

Point these three env vars at **your own host directories** (they can live anywhere — reuse the
`BASE_DIR` from the download step, or set absolute paths directly). `run_with_docker.sh` bind-mounts
them into the container as `DATADIR -> /data`, `MODELDIR -> /model`, `LOGDIR -> /results`; the
in-container side is fixed, so **do not** change `DATA_PATH=/data` in the config or the `/data`,
`/model` paths in the yaml. Ensure `$LOGDIR` is writable (`sudo chmod -R 777 $LOGDIR`).

```bash
export BASE_DIR=/mnt/nvme/mlperf_llama31_8b     # same directory you downloaded into
export DATADIR=${BASE_DIR}/data
export MODELDIR=${BASE_DIR}/model
export LOGDIR=${BASE_DIR}/results
mkdir -p "${LOGDIR}" && sudo chmod -R 777 "${LOGDIR}"
export CONT=rocm/amd-mlperf:llama31_8b_training_6.0
```

### Set Configuration

Set appropriate configuration and system-specific hyperparameters:\
MI308X configuration is in `config_MI308X_1x8x1.sh`\
MI325X configuration is in `config_MI325X_1x8x1.sh`

Both use FP8 hybrid and share the same `conf/llama3.1_8B-pretrain-fp8.yaml`; they differ only in
platform label (and may differ in LR/batch after tuning).

**Run length and logging (defaults tuned for enablement, not a full submission).**
- `PRIMUS_TRAIN_ITERS` defaults to `200` — a short performance / bring-up run (~a few minutes on
  8 GPUs). Set it to `1200000` for the full training run used by the MI350X submission (very long
  on a single 8-GPU node — it targets log perplexity 3.3), or e.g. `50` for a quick smoke test.
- `log_interval` defaults to `10` (via `PRIMUS_LOG_INTERVAL` in the yaml), so a training line is
  printed every 10 steps. The original MI350X submission set this to `9999999` to silence per-step
  logs during the timed run; restore that with `export PRIMUS_LOG_INTERVAL=9999999`.
- `log_throughput` defaults to `true` (via `PRIMUS_LOG_THROUGHPUT`), which adds the TFLOP/s/GPU and
  tokens/s/GPU fields to that line. Set `PRIMUS_LOG_THROUGHPUT=false` to drop them.

**Seeing loss and performance.** Run with `MLPERF_VERBOSE_LOGS=1` to make the per-iteration
training line (loss + throughput) visible:

```bash
source config_MI308X_1x8x1.sh # or: source config_MI325X_1x8x1.sh
export MLPERF_VERBOSE_LOGS=1
export NEXP=1
bash run_with_docker.sh
```

Why this is needed — the original submission silenced the line two ways, and verbose mode undoes
both:
1. **loguru sink level.** Primus emits the training line at **INFO** via loguru, but the submission
   set `stderr_sink_level: ERROR`, dropping it at the source. The yaml now defaults
   `stderr_sink_level` to `INFO` (`PRIMUS_STDERR_SINK_LEVEL` to override).
2. **stderr redirection.** In quiet mode `src/_log_suppression.py` sends stderr to `/dev/null` and
   `run_and_time.sh` adds `2>/dev/null`; the training line goes to stderr, so it is dropped.
   `MLPERF_VERBOSE_LOGS=1` skips the in-process suppression **and** `run_and_time.sh` keeps stderr
   in that mode, so the line survives. (This leaves `_log_suppression.py` unchanged from upstream.)

Note: verbose mode is chattier — framework INFO logs come through too; `grep` for `iteration` to
pull just the training lines. Each looks like:

```
iteration   50/  200 | consumed samples: 1600 | elapsed time per iteration (ms): 1650.0/1650.0 | lm loss: 7.42E+00 | ... | throughput per GPU (TFLOP/s/GPU): 250.0/250.0 | tokens per GPU (tokens/s/GPU): 4950.0/4950.0 |
```

Reading throughput (fields are **per GPU**; the `a/b` form is `current/running-average`):
- Whole-node tokens/s = `tokens/s/GPU * world_size` (e.g. `× 8`).
- Or derive from timing: `global_batch_size * seq_length / (elapsed_time_per_iteration_s)` tokens/s for the whole node.
- Skip the first few iterations (compile + warmup) and read the running-average column.

The MLPerf logger separately records an `overall_throughput` (samples/s over the whole run) in
`$LOGDIR/mlperf_logging.out`; multiply by `seq_length` (8192) for tokens/s.

**Precision provenance.** The FP8 settings in the yaml are not guessed — they are taken from
Primus' own MLPerf FP8 reference config
[`examples/mlperf/configs/MI355X/llama3.1_8B-pretrain-FP8.yaml`](https://github.com/AMD-AGI/Primus/blob/d53c428944c3d74c41c4c96fa2c1776d7e966538/examples/mlperf/configs/MI355X/llama3.1_8B-pretrain-FP8.yaml)
(the FP8 predecessor of AMD's FP4 submission), pinned at the same Primus commit `d53c428` the
Dockerfile builds. Specifically:
- `fp8: hybrid` — FP8 format; scaling recipe defaults to `delayed` (no TE-version/env gate).
- `fp8_amax_history_len: 4` and `fp8_amax_compute_algo: "most_recent"` — MLPerf-tuned amax
  settings, overriding the `trainer_base.yaml` defaults (`1024` / `"max"`).

**BF16 alternative.** A `conf/llama3.1_8B-pretrain-bf16.yaml` is provided for a BF16 baseline (same
config with the fp8 keys omitted; Primus defaults to `bf16: true`). Use it when you want to compare
numerics/throughput against FP8, or to sidestep FP8-specific issues, by overriding `EXP` after
sourcing the config:

```bash
source config_MI308X_1x8x1.sh   # or: source config_MI325X_1x8x1.sh
export EXP=/workspace/code/conf/llama3.1_8B-pretrain-bf16.yaml  # BF16 instead of FP8
export MLPERF_VERBOSE_LOGS=1
export NEXP=1
bash run_with_docker.sh
```

To debug NaN/Inf, set `check_for_nan_in_loss_and_grad: true` (and optionally
`check_for_spiky_loss` / `check_for_large_grads`) in the yaml — training then aborts and reports
where the first NaN/Inf appears (loss vs grad) instead of silently continuing.

### Launch 1 Training Run
If you want to perform a single run, use:
```bash
export NEXP=1
bash run_with_docker.sh
```

### Launch 10 Training Run [Optional]
If you would like to prepare for 10 run submisision, use:

```bash
export NEXP=10
bash run_with_docker.sh
```
After completion, the logs will be available under the directory `$LOGDIR`.

Note:To optimize the machine's performance, the training script will also execute `runtime_tunables.sh` script before any training run.


# 4. Check Quality
### Quality metric

Validation loss

### Quality target

Validation log perplexity = 3.3

### Evaluation frequency

We perform evaluation every **12288** sequences. 

### Evaluation thoroughness

We evaluate using **1024** sequences from our customized validation dataset.

# 5. Background: why this CDNA3 directory exists

## The official MI350X/MI355X submission trains in FP4

This directory is a CDNA3 adaptation of the official submission, which trains in **FP4/mxfp4** — a
CDNA4 (gfx950) feature. Evidence from the original `MI350X_EPYC_9575F_primus/`:

| # | Evidence | Location |
|---|---|---|
| 1 | `fp4: true` / `fp4_recipe: mxfp4` | `conf/llama3.1_8B-pretrain-fp4.yaml` |
| 2 | `FP4=true` / `FP4_RECIPE=mxfp4` | `config_MI350X_1x8x1.sh` |
| 3 | `MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR='mxfp4'` (MLPerf closed declaration) | `config_MI350X_1x8x1.sh` |
| 4 | FP4 GEMM kernel `_ZN5aiter42f4gemm_bf16_per1x32Fp4_...` (A4W4, MXFP4 1×32 block scaling) | `a4w4_tuned_gemms.csv` |
| 5 | `NVTE_MXFP4_USE_HADAMARD=1` | `config_MI350X_1x8x1.sh` |

The aiter FP4 GEMM in (4) needs native FP4 matrix instructions that **CDNA3 (gfx942) does not have**,
which is why MI308X/MI325X cannot reproduce the FP4 submission and this FP8 variant exists.

## MI308X vs MI325X

Same arch (gfx942), same FP8 support, no FP4 on either. They differ only in VRAM
(MI308X ≈ 192 GB, MI325X = 256 GB) and compute. So they share one Dockerfile / yaml / run scripts;
only the config label differs (`config_MI308X_1x8x1.sh` vs `config_MI325X_1x8x1.sh`).

## FP8 config keys (verified against Primus source @ `d53c428`)

- `trainer_base.yaml`: `fp8` (format: `e4m3`/`hybrid`) and `fp8_recipe` (scaling:
  `delayed`/`tensorwise`/`blockwise`/`mxfp8`, default `delayed`) are **two separate fields**.
- `fp8_utils.py`: `fp8` only accepts `"e4m3"`/`"hybrid"` — passing `true` raises `ValueError`.
  Hence the yaml uses `fp8: hybrid` (a single line), not `fp8: true`.
- The `FP8_*` env vars in the config are informational; Primus does not read them to set the recipe.

## Notes / risks

- **Convergence is not guaranteed.** LR is inherited as `8e-4` (tuned for FP4); FP8 may need a
  different value. Watch loss on a short run first.
- **VRAM.** MI308X (~192 GB) is smaller than the MI350X (288 GB) this was tuned on (see the VRAM
  breakdown below). If you hit OOM, lower
  `PRIMUS_MICRO_BATCH_SIZE` from `2` to `1` (halves activation memory; `PRIMUS_GLOBAL_BATCH_SIZE`
  stays `32`, so convergence is unaffected — grad-accumulation just goes 2 → 4). Do **not** change
  `PRIMUS_GLOBAL_BATCH_SIZE`.
- **mxfp8 recipe** (if ever tried) requires `NVTE_ROCM_ENABLE_MXFP8=1` and TE ≥ 2.1; the default
  `delayed` recipe used here has no such gate.
- **Not a valid MLPerf closed submission** — precision differs from the ruleset.

## Parallelism

`TP=PP=CP=EP=1`, pure data-parallel over 8 GPUs (8B fits per GPU), `use_distributed_optimizer=true`
shards optimizer state across DP. `micro_batch_size=2`, `global_batch_size=32`, `seq_length=8192`.

## VRAM breakdown

The estimate below suggests per-GPU usage lands comfortably under MI308X's ~192 GB with the default
`mbs=2`, so no batch-size tuning should be needed. Treat these as rough figures — **check actual
usage on your run** (e.g. `amd-smi monitor` / `rocm-smi`), and if a GPU approaches its limit, lower
`PRIMUS_MICRO_BATCH_SIZE` (see the OOM note under "Notes / risks"). Rough per-GPU budget for
8B + FP8 + `mbs=2`, `seq=8192`, DP=8:

| Component | Size | Scales with | Sharded by DP? |
|---|---|---|---|
| Weights (BF16 working copy) | ~16 GB | param count | No (needed for local fwd/bwd) |
| Gradients (BF16) | ~16 GB | param count | No (ZeRO-1 doesn't shard grads) |
| Optimizer state (fp32 master + Adam m + v = 12 B/param) | 96 GB / 8 ≈ **12 GB** | param count | **Yes** — `use_distributed_optimizer` shards it across the 8 DP ranks |
| Activations (no recompute) | ~60–70 GB | `layers × seq × hidden × mbs` | No (local to each rank) |
| Temp buffers + fragmentation | ~15–25 GB | — | — |

Note: FP8 training does **not** save VRAM here — it speeds up matmuls/comms but still keeps BF16
master weights and adds FP8/scale copies, so total memory is roughly the same as BF16 (or slightly
less). VRAM is dominated by activations + weights + optimizer state, not the compute precision.

Key points:
- **Optimizer state depends on parameter count, not batch size** (`8B × 12 B = 96 GB`), and is the
  only piece ZeRO-1 shards (`/8`), because in DP all 8 ranks would otherwise hold identical copies.
  Weights/gradients stay full-size because each rank runs a complete forward/backward locally.
- **Activations are the big, batch-dependent term.** Training must keep every layer's forward
  activations until backprop consumes them (unlike inference, which can reuse buffers), so memory
  grows with `layers × seq_length × mbs`. `seq_length=8192` is what makes this large.
- **To reduce memory** you would enable activation recomputation (`recompute_granularity: selective`),
  trading ~30% compute to drop activations — deliberately *off* here since 192 GB fits comfortably.
- With this much headroom, **leave `micro_batch_size=2` as-is**; only drop to 1 if a future change
  (longer seq, larger model) pushes you toward OOM.
