# Two-Node CDNA3 Llama 3.1 8B Training (16 GPUs)

This guide describes the scheduler-free, two-node launch path for the Llama 3.1 8B
implementation. It uses two homogeneous, supported CDNA3 nodes with eight GPUs per node and starts
one `torchrun` launcher locally on node0 and another over SSH on node1. The platform is selected
explicitly through `DGXSYSTEM_2N`; the launcher has no hardware-specific default.

Complete the image, dataset, and tokenizer setup in the [implementation README](../README.md)
before using this launcher.

FP8 is the default two-node recipe, consistent with the single-node launch convention; BF16 is
available as an optional baseline. The checked-in two-node FP8 parameters match the configuration
observed to reach the target on MI325X. These enablement recipes are not a claim that this precision path is an
official MLPerf closed-submission recipe.

## Configuration overview

| Setting | Value |
|---|---:|
| Nodes | 2 |
| GPUs per node | 8 |
| World size | 16 |
| Parallelism | Data parallelism (TP=PP=CP=EP=1) |
| Micro-batch size | 2 |
| Global batch size | 32 by default (`PRIMUS_GLOBAL_BATCH_SIZE` overrides it) |
| Gradient accumulation | 1 by default (GBS32 / MBS2 / DP16) |

Select one of the platform configurations before launch:

| Platform | `DGXSYSTEM_2N` | Status |
|---|---|---|
| MI325X | `MI325X_2x8x1` | Corresponding BF16 and E4M3 FP8 enablement runs reached the target |
| MI308X | `MI308X_2x8x1` | Uses the shared recipes of MI325X |

The multi-node path consists of:

- `config_MI308X_2x8x1.sh` / `config_MI325X_2x8x1.sh`: platform selectors;
- `config_common_2x8x1.sh`: shared two-node batch and network overlay on each platform's
  single-node configuration;
- `conf/llama3.1_8B-pretrain-fp8.yaml`: default E4M3 + tensorwise/current FP8 recipe;
- `conf/llama3.1_8B-pretrain-bf16.yaml`: optional two-node BF16 baseline;
- `run_with_docker_2node.sh`: node0 launcher and node1 SSH coordination;
- `../run_with_docker.sh`: shared per-node training container launcher;
- `bnxt_rdma_overlay.sh`: optional Broadcom RDMA userspace-library discovery and mounts.

Multi-node jobs use torchrun's static rendezvous. Node 0 hosts the rendezvous endpoint at
`NODE0_IP:MASTER_PORT`; node 0 uses `NODE_RANK=0` and node 1 uses `NODE_RANK=1`.

## Prerequisites

- Two homogeneous nodes matching one of the supported platform configurations, each with eight
  available GPUs.
- The same training image and tag on both nodes.
- Bash 5.1 or newer on node0 and Bash 4.3 or newer as the remote login shell on node1.
- Passwordless SSH from node0 to node1.
- Direct Docker daemon access for the node0 user and `SSH_USER`; `docker info` must succeed without
  `sudo` or a password prompt.
- The repository, dataset, tokenizer, and results directories at identical absolute paths on both
  Docker daemon hosts. Identical paths do not imply shared storage; separately staged node-local
  datasets must also contain the generated Megatron dataset cache described below.
- TCP connectivity from node1 to `NODE0_IP:MASTER_PORT` (default port: `29502`), with the port
  available on node0.
- A working cross-node RoCE network when RDMA transport is enabled.

If the launcher runs inside a development container, use host networking and expose the host
Docker socket plus the relevant network/RDMA sysfs devices. The launcher must see the same network
interface names as the training container; otherwise, set `NCCL_SOCKET_IFNAME` and
`GLOO_SOCKET_IFNAME` explicitly. Docker bind-mount source paths are resolved by the daemon host,
not by the launcher container.

### Configure passwordless SSH

Run these commands on node0 as the same user and in the same environment that will run
`run_with_docker_2node.sh`. Only node0-to-node1 passwordless access is required by this launcher.

```bash
export NODE1_IP=192.0.2.11      # documentation address; replace with the node1 address
export SSH_USER=training        # replace with the node1 login user
export SSH_PORT=22

install -d -m 700 "${HOME}/.ssh"
if [[ ! -f "${HOME}/.ssh/id_ed25519" ]]; then
  ssh-keygen -t ed25519 -N '' -f "${HOME}/.ssh/id_ed25519" \
    -C "mlperf-2node@$(hostname)"
fi
[[ -f "${HOME}/.ssh/id_ed25519.pub" ]] || \
  ssh-keygen -y -f "${HOME}/.ssh/id_ed25519" > "${HOME}/.ssh/id_ed25519.pub"

cat "${HOME}/.ssh/id_ed25519.pub"
```

Copy the printed public key to node1 and append it as one line to the remote user's
`authorized_keys`:

```bash
# Run on node1 as SSH_USER.
install -d -m 700 "${HOME}/.ssh"
touch "${HOME}/.ssh/authorized_keys"
chmod 600 "${HOME}/.ssh/authorized_keys"
# Append the copied node0 public key as one complete line to this file.
```

Then verify from node0 that SSH is non-interactive and that the remote user can access Docker:

```bash
ssh -p "${SSH_PORT}" \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=10 \
  "${SSH_USER}@${NODE1_IP}" \
  'docker info >/dev/null && hostname'
```

`BatchMode=yes` must succeed without a password or passphrase prompt. If the launcher runs inside a
development container, make the same SSH identity and `known_hosts` available there.

### Dataset cache on node-local storage

The downloaded dataset contains the four preprocessed `.bin` and `.idx` files. Megatron creates
additional GPT dataset indices at runtime under the following directories:

```text
${DATADIR}/c4-train.en_6_text_document/cache/GPTDataset_indices/
${DATADIR}/c4-validation-91205-samples.en_text_document/cache/GPTDataset_indices/
```

During a distributed run, global rank 0 creates a missing cache and the other ranks load it after a
barrier. **If `DATADIR` is on storage shared by both nodes, no manual cache copy is needed. If each
node instead has an independent local copy at the same absolute path, files created by rank 0 on
node0 are not visible on node1 and must be copied manually.** In that case, node1 can fail with a
path such as:

```text
/data/c4-train.en_6_text_document/cache/GPTDataset_indices/...-document_index.npy
```

After node0 has generated the cache, stop the failed run and copy both complete cache directories
from node0 to node1. Run the following on node0; it does not copy the large `.bin` or `.idx` files:

```bash
export DATADIR=/path/to/mlperf_data/data
export NODE1_IP=192.0.2.11
export SSH_USER=training
export SSH_PORT=22

for dataset in \
  c4-train.en_6_text_document \
  c4-validation-91205-samples.en_text_document
do
  cache="${DATADIR}/${dataset}/cache/GPTDataset_indices"
  test -d "${cache}" || {
    echo "Missing node0 cache: ${cache}" >&2
    exit 1
  }

  ssh -p "${SSH_PORT}" "${SSH_USER}@${NODE1_IP}" \
    "mkdir -p '${cache}'"
  scp -P "${SSH_PORT}" -p "${cache}/"* \
    "${SSH_USER}@${NODE1_IP}:${cache}/"
done
```

Copy the entire directory rather than only the filename reported by the exception. Each cache key
contains a description plus document, sample, and shuffle indices. Repeat the copy when a changed
dataset configuration produces a new cache key.

## Launch a validation run

Run the launcher from node0:

```bash
cd /path/to/MI308X_MI325X_CDNA3_primus/2nodes

# Platform
export DGXSYSTEM_2N=MI325X_2x8x1  # use MI308X_2x8x1 on MI308X

# Network topology
export NODE0_IP=192.0.2.10      # documentation address; replace with the node0 address
export NODE1_IP=192.0.2.11      # documentation address; replace with the node1 address
export SSH_USER=training
export SSH_PORT=22
export MASTER_PORT=29502

# Identical image and host paths on both nodes
export CONT=rocm/amd-mlperf:llama31_8b_training_6.0
export DATADIR=/mnt/shared/mlperf/llama31_8b/data
export MODELDIR=/mnt/shared/mlperf/llama31_8b/model
export LOGDIR=/mnt/shared/mlperf/llama31_8b/results

# Reproducible short run
export SEED=1234
export NEXP=1
export PRIMUS_TRAIN_ITERS=50
export PRIMUS_LR=3e-4
export PRIMUS_MIN_LR=3e-5
export MLPERF_VERBOSE_LOGS=1

# FP8 (E4M3 + tensorwise/current, CK v3 BF16 conversion 2/RTZ) is selected by default.
bash run_with_docker_2node.sh
```

Each platform wrapper inherits its matching single-node kernel settings and applies the same
two-node batch, recipe, and network defaults. On a new platform, confirm the `[config]` and `[rdma]`
lines in both node logs select the expected NIC, HCA list, GID index, and provider.

The launcher defaults to 50 iterations. Both platforms default to GBS32, LR `3e-4` / minimum LR
`3e-5`, and the following recipes:

| Mode | Experiment YAML | Synthetic warmup | CK v3 BF16 conversion | Notes |
|---|---|---|---|---|
| FP8 (default) | `/workspace/code/2nodes/conf/llama3.1_8B-pretrain-fp8.yaml` | `fp8_hybrid` | `2` (RTZ) | E4M3 forward + backward, tensorwise/current scaling, three FP32 settings, and collective AVG |
| BF16 (optional) | `/workspace/code/2nodes/conf/llama3.1_8B-pretrain-bf16.yaml` | `bf16` | `0` (RTNE) | BF16 model path with the same three FP32 settings and collective AVG |

With no precision override, the launcher selects the FP8 YAML and infers the `fp8_hybrid`
synthetic warmup. To select the optional BF16 baseline, set these before launching:

```bash
export EXP=/workspace/code/2nodes/conf/llama3.1_8B-pretrain-bf16.yaml
# Use RTNE (round to nearest, ties to even) for CK FA v3 float-to-BF16 conversion on gfx942.
export NVTE_CK_HOW_V3_BF16_CVT=0
export WARMUP_RECIPE=bf16
```

Set `PRIMUS_TRAIN_ITERS=1200000` for full training. `PRIMUS_GLOBAL_BATCH_SIZE` can override GBS32;
the launcher forwards it to both nodes and recomputes `PRIMUS_EVAL_INTERVAL`. Convergence evidence
currently covers GBS32. An override must be a positive multiple of DP×MBS (`16×2=32`) and must
divide the 12,288-sample evaluation interval exactly.

## Launcher options

| Variable | Default | Description |
|---|---|---|
| `SSH_USER` | `root` | Node1 login user |
| `SSH_PORT` | `22` | Node1 SSH port |
| `MASTER_PORT` | `29502` | Torchrun rendezvous port on node0 |
| `DGXSYSTEM_2N` | required | Configuration filename suffix selected from the platform table |
| `PRIMUS_TRAIN_ITERS` | `50` | Per-run iteration cap |
| `PRIMUS_GLOBAL_BATCH_SIZE` | `32` | Global batch size forwarded identically to both nodes |
| `PRIMUS_LR` | `3e-4` | Peak learning rate; override together with `PRIMUS_MIN_LR` |
| `PRIMUS_MIN_LR` | `3e-5` | Minimum learning rate; override together with `PRIMUS_LR` |
| `PRIMUS_LR_WARMUP_ITERS` | `64` | Learning-rate warmup iterations forwarded identically to both nodes |
| `EXP` | two-node FP8 YAML | Experiment-YAML selection; use the two-node BF16 YAML for the optional BF16 baseline |
| `WARMUP_RECIPE` | inferred from `EXP` | `bf16` for the standard BF16 YAML, `fp8_hybrid` for the standard FP8 YAML; required for custom YAML names |
| `NVTE_CK_HOW_V3_BF16_CVT` | inferred from `EXP` | `2`/RTZ for standard FP8 and `0`/RTNE for standard BF16; explicit `0`, `1`, or `2` overrides both nodes |
| `MLPERF_VERBOSE_LOGS` | `1` | Enable per-iteration training output |
| `RUN_ID` | generated | Suffix for container and launcher-log names |
| `LOG_PREFIX` | `run_2node` | Launcher-log filename prefix |

`NODE0_IP`, `NODE1_IP`, `CONT`, `DATADIR`, `MODELDIR`, `LOGDIR`, and `SEED` are required.

## Network configuration

The selected `config_*_2x8x1.sh` is sourced independently on each node and selects the following
values:

| Variable | Default selection |
|---|---|
| `NCCL_SOCKET_IFNAME` | Interface carrying the default route (`iproute2` or `/proc/net/route`) |
| `GLOO_SOCKET_IFNAME` | Same interface as `NCCL_SOCKET_IFNAME` |
| `NCCL_IB_HCA` | Broadcom PCI vendor `0x14e4`, then `mlx5*`, then any InfiniBand device |
| `NCCL_IB_GID_INDEX` | RoCE v2 entry on the first detected HCA; fallback `3` |
| `NCCL_NET_GDR_LEVEL` | `3` |

These variables can be exported before starting the launcher. A caller override is applied to both
nodes, so an interface name, HCA name, or GID index must be valid on both systems.
When node-local names differ, leave the variable unset and use per-node auto-detection.
If a node has no default route, set both `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` explicitly;
the configuration fails early rather than guessing an arbitrary UP interface.

Set `NCCL_DEBUG=INFO` for communication diagnostics. Set `NCCL_IB_DISABLE=1` only for a
socket-transport diagnostic; it also disables the Broadcom userspace overlay described below.

## Broadcom RDMA userspace overlay

On AMD multi-node systems with a Broadcom RDMA device, `run_with_docker.sh` automatically calls
`bnxt_rdma_overlay.sh`. The helper identifies the HCA by PCI vendor ID, so names such as
`bnxt_re0` and `rdma0` are both supported. It inspects the Docker daemon host and selects a unique
`libibverbs.so.1` and `libbnxt_re-rdmavN.so` pair that matches the running `bnxt_re` kernel module
and private ABI. It then mounts those libraries read-only into the training container.

This mechanism does not replace the host kernel driver, install packages, or run an additional
communication preflight. If no unique compatible pair is found, container startup fails with a
discovery error.

Auto-discovery is recommended. If both nodes use identical host library paths, they can be
overridden as a pair:

```bash
export BNXT_RDMA_LIBIBVERBS_HOST_PATH=/usr/lib/x86_64-linux-gnu/libibverbs.so.1
export BNXT_RDMA_PROVIDER_HOST_PATH=/usr/local/lib/libbnxt_re-rdmav34.so
```

## Logs and failure handling

The launcher writes two per-node execution logs in the implementation root unless `LOG_PREFIX`
contains another path:

- `${LOG_PREFIX}_${RUN_ID}_node0.log`
- `${LOG_PREFIX}_${RUN_ID}_node1.log`

Training and MLPerf logs are written under `LOGDIR` on each node. If `LOGDIR` is node-local, inspect
both hosts. If it is shared storage, both nodes write into the same directory.

If either node launcher exits with a nonzero status, the supervisor attempts `docker stop` followed
by `docker rm -f` for both run-specific container names and returns the failing status. HUP, INT,
and TERM use the same cleanup path. Remote cleanup is best-effort when node1 is unreachable; in
that case, connect to node1 and remove the container name printed by the launcher.

## Troubleshooting

| Symptom | Check |
|---|---|
| SSH preflight prompts or fails | Re-run the `BatchMode=yes` check; verify the selected public key, remote `~/.ssh` permissions, and server-side public-key authentication. |
| Image preflight fails | Run `docker info` and `docker image inspect "$CONT"` as the launch user on both nodes. |
| Repository preflight fails | Confirm `REPO_DIR` exists at the same absolute path on node1. |
| `GPTDataset_indices` file is missing on node1 | If `DATADIR` is node-local, copy both generated cache directories from node0 as described above. |
| Rendezvous times out | Verify `NODE0_IP`, firewall rules, and node1-to-node0 access to `MASTER_PORT`. |
| Gloo connection fails | Verify `GLOO_SOCKET_IFNAME` resolves to a node1-to-node0 reachable interface on both nodes. |
| NCCL or RDMA initialization fails | Check the HCA, GID index, and discovery messages in both logs. |
| Training hangs at the first collective | Enable `NCCL_DEBUG=INFO`; compare with a socket-only diagnostic. |

The launcher coordinates exactly two nodes. It does not build or distribute images, synchronize
repository contents, or integrate with a cluster scheduler.
