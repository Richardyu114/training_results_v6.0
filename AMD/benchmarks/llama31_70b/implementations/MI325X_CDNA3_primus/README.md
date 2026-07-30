# Llama 3.1 70B pretraining on MI325X (CDNA3, Primus)

Pretraining performance run on 2 x 8 MI325X. FP16 and BF16 recipes, both shipped.

| | |
|---|---|
| Model | Llama 3.1 70B (80 layers, hidden 8192, FFN 28672, 64 heads / 8 KV groups) |
| Precision | FP16 with dynamic loss scaling, or BF16 (no scaling) |
| Parallelism | TP=2, PP=8, CP=1 -> DP=1 on 16 GPUs (overridable, see 4.2) |
| Batch | micro 2, global 64 (32 gradient-accumulation steps) |
| Sequence length | 1024 |
| LR | 1e-5 -> 1e-6 cosine, 10% warmup |
| Data | preprocessed C4, same corpus as llama31_8b |
| Goal | throughput (TFLOP/s/GPU, tokens/s/GPU) over a few hundred steps |

This is a performance/enablement recipe, not a convergence run and not an MLPerf
submission: no llama31_70b MLPerf benchmark exists, so there is no quality target and
the compliance checker is off by default.

---

## 1. Prerequisites

- Two MI325X nodes, 8 GPUs each, ROCm host stack installed, Docker usable without a
  password prompt.
- Passwordless SSH from node0 to node1 as the same user.
- RDMA (RoCE v2) between the two nodes.
- This directory present at the **same absolute path on both nodes**.
- Dataset and tokenizer on both nodes (see step 3).

## 2. Build the image

Run on **both** nodes, from this directory. Tag identically on both.

```bash
docker build --build-arg GPU_ARCHS=gfx942 -t llama31_70b:mi325x .
```

`GPU_ARCHS=gfx942` builds TransformerEngine and AITER for CDNA3 only, which roughly
halves build time versus the default `gfx950;gfx942`.

Alternatively build once and copy:

```bash
# node0
docker save llama31_70b:mi325x | gzip > llama31_70b_mi325x.tar.gz
scp llama31_70b_mi325x.tar.gz <node1>:/tmp/
# node1
gunzip -c /tmp/llama31_70b_mi325x.tar.gz | docker load
```

## 3. Download data and tokenizer

This recipe reuses the llama31_8b assets **unchanged**. The Llama 3.1 family shares one
tokenizer (vocab 128256), so the C4 corpus tokenized for 8B is directly valid for 70B, and
128256 is divisible by `TP`, so no vocab padding is needed. If you already have
these staged for an 8B run, skip to step 4 and point `DATADIR` / `MODELDIR` at them.

Download **on the host, not inside the container** — the image contains only code. Pick any
host directory; only the in-container paths `/data` and `/model` are fixed. Do this on
**both** nodes (or download once and copy).

```bash
export BASE_DIR=/mnt/nvme/llama31_70b     # <-- any path you like, same on both nodes
mkdir -p "${BASE_DIR}" && cd "${BASE_DIR}"

# data (~80 GB) -> creates ./data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# tokenizer / model config (~30 GB) -> creates ./model
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

Verify on **both** nodes — each dataset partition is a `.bin` + `.idx` pair:

```bash
ls "${BASE_DIR}"/data/c4-train.en_6_text_document.{bin,idx}
ls "${BASE_DIR}"/data/c4-validation-91205-samples.en_text_document.{bin,idx}
ls "${BASE_DIR}"/model/tokenizer.json "${BASE_DIR}"/model/config.json
```

Those two document prefixes are what `train_data_path` and `valid_data_path` in the yaml
resolve to inside the container. Source corpus:
[allenai/c4](https://huggingface.co/datasets/allenai/c4) (`en/3.0.1`). For the preprocessing
background and these same steps in their original context, see the llama31_8b
implementation:
[`../../../llama31_8b/implementations/MI308X_MI325X_CDNA3_primus/README.md`](../../../llama31_8b/implementations/MI308X_MI325X_CDNA3_primus/README.md).

## 4. Run

Run **only on node0**; it starts node1 over SSH.

```bash
export NODE0_IP=<node0 IP>
export NODE1_IP=<node1 IP>
export CONT=llama31_70b:mi325x
export BASE_DIR=/mnt/nvme/llama31_70b     # same directory you downloaded into
export DATADIR=${BASE_DIR}/data
export MODELDIR=${BASE_DIR}/model
export LOGDIR=${BASE_DIR}/results
export SEED=1234
export PRIMUS_TRAIN_ITERS=100

mkdir -p "$LOGDIR" && chmod -R 777 "$LOGDIR"
bash run_with_docker_2node.sh
```

Launcher defaults, override by exporting before the call: `MASTER_PORT=29502`, `SSH_USER=root`, `SSH_PORT=22`,
`EXP=/workspace/code/conf/llama3.1_70B-pretrain-fp16.yaml`, `PRIMUS_GLOBAL_BATCH_SIZE=64`,
`PRIMUS_LR=1e-5`, `PRIMUS_MIN_LR=1e-6`, `PRIMUS_LR_WARMUP_FRACTION=0.1`,
`MLPERF_VERBOSE_LOGS=1`.

For a quick functional check, use the same command with `PRIMUS_TRAIN_ITERS=30`.

### 4.1 BF16 instead of FP16

The two recipes differ **only** in the compute dtype — same parallelism, batch, schedule,
data and kernels. Switch by pointing `EXP` at the other yaml:

```bash
export EXP=/workspace/code/conf/llama3.1_70B-pretrain-bf16.yaml
```

`run_and_time.sh` derives the precision from the filename (`*fp16*.yaml` / `*bf16*.yaml`)
and aborts if it matches neither, so keep that naming.

### 4.2 Changing the parallel layout

`PRIMUS_TENSOR_PARALLEL_SIZE` and `PRIMUS_PIPELINE_PARALLEL_SIZE` are read from the
environment by `config_MI325X_2x8x1.sh` (defaults 2 and 8), so a different layout needs no
file edit:

```bash
PRIMUS_TENSOR_PARALLEL_SIZE=1 PRIMUS_PIPELINE_PARALLEL_SIZE=4 bash run_with_docker_2node.sh
```

DP is derived, not set: `DP = world_size / (TP x PP)`. The config validates at startup that
`world_size % (TP x PP) == 0` and `GBS % (DP x MBS) == 0`, prints the resolved
`DP / grad_accum`, and warns when `grad_accum < PP`. Constraints:

- `80 % PP == 0` — PP must divide the layer count, so PP ∈ {1, 2, 4, 8}.
- Megatron requires `TP > 1` for `sequence_parallel`; at TP=1 it must be turned off in the
  yaml or startup asserts.
- `TP x PP >= 4` on 16 GPUs, otherwise the model does not fit (see the memory note below).

## 5. Read the results

Logs land in `$LOGDIR`:

| File | Contents |
|---|---|
| `..._node0.log` | launcher output, MLLOG events, `RESULT,LLAMA3.1_70B,...` line |
| `..._node1.log` | **per-iteration loss / TFLOP/s / memory lines** |
| `mlperf_logging.out` | MLLOG event stream |

**FP16 skipped iterations at the start are normal**: expect 8–10 while the dynamic loss
scale settles down from 2^32, with no `lm loss` printed for those. BF16 shows 0.
`number of nan iterations` should be 0 either way; anything else is a real failure.

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | ROCm + TransformerEngine + AITER + Primus image |
| `conf/llama3.1_70B-pretrain-fp16.yaml` | model, parallelism, batch, FP16, data paths |
| `conf/llama3.1_70B-pretrain-bf16.yaml` | same, BF16; differs only in the precision block |
| `config_MI325X_2x8x1.sh` | two-node env, RoCE auto-discovery, batch/parallelism validation |
| `run_with_docker_2node.sh` | node0 entry point; drives both nodes |
| `run_with_docker.sh` | starts the container on one node and runs the training |
| `run_and_time.sh` | in-container torchrun launch and timing |
| `bnxt_rdma_overlay.sh` | mounts host Broadcom RDMA userspace libraries |
| `runtime_tunables.sh` | optional host tuning before a measurement run (needs sudo) |
| `precompile_aiter.py` | build-time AITER JIT prewarm (RoPE + enum modules) |
| `requirements.txt` | mlcommons logging 6.0.0-rc5 |
| `primus_mllog-0.1.21-py3-none-any.whl` | MLLOG event emitter used by Primus |
| `src/train.py` | Primus trainer entry point |
| `src/_log_suppression.py` | quiet-mode stdout filter (`MLPERF_VERBOSE_LOGS=0`) |

## Tuning notes

- **Batch.** GBS must be a multiple of `DP x MBS`; the quotient is the gradient-accumulation
  count. At the default DP=1 / MBS=2, GBS=64 gives 32 accumulation steps.
- **Pipeline bubble.** PP=8 with 32 accumulation steps runs `32 + 8 - 1 = 39` pipeline slots
  for 32 useful microbatches, i.e. ~82% pipeline efficiency. Raising GBS raises that ratio;
  account for it when comparing TFLOP/s against single-stage (PP=1) numbers.
- **LR schedule is self-contained.** `lr_decay_iters` is tied to `train_iters`, so the cosine
  completes within whatever run length you pick and `min_lr` is actually reached.
- **Warmup is derived, not hardcoded.** The yaml sets `lr_warmup_fraction: 0.1` and
  `lr_decay_iters: ${PRIMUS_TRAIN_ITERS}`, so Megatron computes the warmup step count as
  `0.1 x train_iters` — 200 iters gives 20, 1000 gives 100, no edit needed when the run
  length changes. Override the ratio with `PRIMUS_LR_WARMUP_FRACTION`.
  `PRIMUS_LR_WARMUP_ITERS` is rejected: Megatron asserts that only one of
  `lr_warmup_fraction` / `lr_warmup_iters` may be set. Note the coupling — if you ever pin
  `lr_decay_iters` to a long fixed horizon, the fraction must become a step count too, or
  warmup resolves to a fraction of that horizon instead of of the run.
- **Two warmups, unrelated.** `PRIMUS_LR_WARMUP_FRACTION` (0.1) is the learning-rate ramp and
  is part of training. `SYNTH_WARMUP_STEPS` (5) is this harness's synthetic-data warmup that
  compiles kernels and fills the memory pool before the timed region. `WARMUP_RECIPE` has no
  FP16 value, so it is left empty (no autocast).
- **`NVTE_CK_IS_V3_ATOMIC_FP32=1` is required on gfx942.** With 0, the CK v3 attention
  backward accumulates in non-FP32 atomics and overflows to Inf, producing "found Inf in
  local grad norm" on the first step.
- **Collectives at DP=1.** `use_distributed_optimizer`, `overlap_grad_reduce` and
  `overlap_param_gather` are on, matching the 8B two-node recipe. At DP=1 they are no-ops —
  one optimizer shard, and no gradient all-reduce or parameter all-gather to overlap — but
  keeping them on means the same yaml stays correct at DP > 1. They are the first switches to
  turn off if the run fails during the optimizer or gradient-reduction step.
- **Optimizer offload is off.** `optimizer_cpu_offload: false`; CPU offload would also
  require `use_precision_aware_optimizer`, and there is no memory pressure here.
- **Virtual pipeline (VPP) is off, and `overlap_p2p_comm` with it.**
  `virtual_pipeline_model_parallel_size: null` means each rank owns one contiguous block of
  10 layers. Interleaving would split it into VPP chunks so a microbatch loops the pipeline
  VPP times, shrinking the bubble from `(PP-1)/GA = 21.9%` to `(PP-1)/(VPP x GA) = 10.9%` at
  VPP=2. Megatron requires `num_layers % (PP x VPP) == 0`, so 80 layers at PP=8 admits only
  VPP=1 or 2. Left off for the baseline: it changes scheduling and activation memory at the
  same time. `overlap_p2p_comm` is explicitly `false` because Megatron force-disables it
  without virtual stages (and warns at PP > 1); turning VPP on is what unlocks it, and the
  two should be enabled together.
- **Loss scaling.** FP16's representable range bottoms out near 6e-8, below which small
  gradients flush to zero. Dynamic loss scaling multiplies the loss before backward and
  unscales before the optimizer update, halving the scale on any overflowing step. Left at
  the framework defaults (start 2^32, window 1000, hysteresis 2). Per-iteration throughput is
  unaffected, but a 100-step run spends ~10% of its steps in the settling phase, which is why
  end-to-end and steady-state throughput can differ in either direction. BF16 shares FP32's
  exponent range and needs none of this.
- **FP32 accumulation is kept in both recipes.** `accumulate_allreduce_grads_in_fp32` and
  `attention_softmax_in_fp32` are on for BF16 too. BF16 has 8 mantissa bits against FP16's
  10, so it is *less* precise per value, and reductions and softmax are exactly where that
  shows up.
- **Sequence length.** 1024 here. Changing it requires changing both `seq_length` and
  `max_position_embeddings` in the yaml.
