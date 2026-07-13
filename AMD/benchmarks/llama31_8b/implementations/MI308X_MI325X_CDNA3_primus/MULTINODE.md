# Two-Node MI308X Llama 3.1 8B Training (16 GPUs)

This guide describes the scheduler-free, two-node launch path for the Llama 3.1 8B
implementation. The provided configuration uses two MI308X nodes with eight GPUs per node and
starts one `torchrun` launcher locally on node0 and another over SSH on node1.

Complete the image, dataset, and tokenizer setup in [README.md](README.md) before using this
launcher.

The CDNA3 FP8 configuration is provided for functional enablement. It is not an MLPerf closed
submission configuration because its numerical format differs from the submitted recipe.

## Configuration overview

| Setting | Value |
|---|---:|
| Nodes | 2 |
| GPUs per node | 8 |
| World size | 16 |
| Parallelism | Data parallelism (TP=PP=CP=EP=1) |
| Micro-batch size | 2 |
| Global batch size | 64 |
| Gradient accumulation | 2 |

The multi-node path consists of:

- `config_MI308X_2x8x1.sh`: model, batch, and network configuration;
- `run_with_docker_2node.sh`: node0 launcher and node1 SSH coordination;
- `run_with_docker.sh`: per-node training container launcher;
- `bnxt_rdma_overlay.sh`: optional Broadcom RDMA userspace-library discovery and mounts.

Multi-node jobs use torchrun's static rendezvous. Node 0 hosts the rendezvous endpoint at
`NODE0_IP:MASTER_PORT`; node 0 uses `NODE_RANK=0` and node 1 uses `NODE_RANK=1`.

## Prerequisites

- Two MI308X nodes, each with eight available GPUs.
- The same training image and tag on both nodes.
- Bash 5.1 or newer on node0 and Bash 4.3 or newer as the remote login shell on node1.
- Passwordless SSH from node0 to node1.
- Direct Docker daemon access for the node0 user and `SSH_USER`; `docker info` must succeed without
  `sudo` or a password prompt.
- The repository, dataset, tokenizer, and results directories at identical absolute paths on both
  Docker daemon hosts.
- TCP connectivity from node1 to `NODE0_IP:MASTER_PORT` (default port: `29502`), with the port
  available on node0.
- A working cross-node RoCE network when RDMA transport is enabled.

By default, each node runs `sudo /sbin/sysctl vm.drop_caches=3` before an experiment. Configure
non-interactive sudo on both nodes, or set `CLEAR_CACHES=0`.

If the launcher runs inside a development container, use host networking and expose the host
Docker socket plus the relevant network/RDMA sysfs devices. The launcher must see the same network
interface names as the training container; otherwise, set `NCCL_SOCKET_IFNAME` and
`GLOO_SOCKET_IFNAME` explicitly. Docker bind-mount source paths are resolved by the daemon host,
not by the launcher container.

Before launching, verify node1 access from node0:

```bash
export NODE1_IP=10.0.0.11       # replace with the node1 address
export SSH_USER=training        # replace with the node1 login user
export SSH_PORT=22

ssh -p "${SSH_PORT}" "${SSH_USER}@${NODE1_IP}" \
  'set -e; test -n "$BASH_VERSION"; bash --version | head -1; docker info >/dev/null; hostname'
```

## Launch a validation run

Run the launcher from node0:

```bash
cd /path/to/MI308X_MI325X_CDNA3_primus

# Network topology
export NODE0_IP=10.0.0.10       # address reachable from node1
export NODE1_IP=10.0.0.11
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

bash run_with_docker_2node.sh
```

The launcher defaults to 50 iterations. For longer training, set `PRIMUS_TRAIN_ITERS` explicitly
and use the learning-rate schedule validated for the selected numerical recipe. The global batch
size is 64, so convergence should be evaluated independently from the single-node configuration.

The default experiment is FP8 hybrid. To run the BF16 configuration, add:

```bash
export EXP=/workspace/code/conf/llama3.1_8B-pretrain-bf16.yaml
export WARMUP_RECIPE=bf16
```

## Launcher options

| Variable | Default | Description |
|---|---|---|
| `SSH_USER` | `root` | Node1 login user |
| `SSH_PORT` | `22` | Node1 SSH port |
| `MASTER_PORT` | `29502` | Torchrun rendezvous port on node0 |
| `REPO_DIR` | launcher directory | Repository path shared by both nodes |
| `DGXSYSTEM_2N` | `MI308X_2x8x1` | Configuration filename suffix |
| `PRIMUS_TRAIN_ITERS` | `50` | Per-run iteration cap |
| `MLPERF_VERBOSE_LOGS` | `1` | Enable per-iteration training output |
| `CLEAR_CACHES` | `1` | Drop host page cache before each experiment |
| `RUN_ID` | generated | Suffix for container and launcher-log names |
| `LOG_PREFIX` | `run_2node` | Launcher-log filename prefix |

`NODE0_IP`, `NODE1_IP`, `CONT`, `DATADIR`, `MODELDIR`, `LOGDIR`, and `SEED` are required.

## Network configuration

`config_MI308X_2x8x1.sh` is sourced independently on each node and selects the following values:

| Variable | Default selection |
|---|---|
| `NCCL_SOCKET_IFNAME` | Interface carrying the default route |
| `GLOO_SOCKET_IFNAME` | Same interface as `NCCL_SOCKET_IFNAME` |
| `NCCL_IB_HCA` | `bnxt_re*`, then `mlx5*`, then any InfiniBand device |
| `NCCL_IB_GID_INDEX` | RoCE v2 entry on the first detected HCA; fallback `3` |
| `NCCL_NET_GDR_LEVEL` | `3` |

These variables can be exported before starting the launcher. A caller override is applied to both
nodes, so an interface name, HCA name, or GID index must be valid on both systems.
When node-local names differ, leave the variable unset and use per-node auto-detection.

Set `NCCL_DEBUG=INFO` for communication diagnostics. Set `NCCL_IB_DISABLE=1` only for a
socket-transport diagnostic; it also disables the Broadcom userspace overlay described below.

## Broadcom RDMA userspace overlay

On AMD multi-node systems with `bnxt_re` devices, `run_with_docker.sh` automatically calls
`bnxt_rdma_overlay.sh`. The helper inspects the Docker daemon host and selects a unique
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

The launcher writes two per-node execution logs in the script directory unless `LOG_PREFIX`
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
| Image preflight fails | Run `docker info` and `docker image inspect "$CONT"` as the launch user on both nodes. |
| Repository preflight fails | Confirm `REPO_DIR` exists at the same absolute path on node1. |
| Rendezvous times out | Verify `NODE0_IP`, firewall rules, and node1-to-node0 access to `MASTER_PORT`. |
| Gloo connection fails | Verify `GLOO_SOCKET_IFNAME` resolves to a node1-to-node0 reachable interface on both nodes. |
| NCCL or RDMA initialization fails | Check the HCA, GID index, and discovery messages in both logs. |
| Training hangs at the first collective | Enable `NCCL_DEBUG=INFO`; compare with a socket-only diagnostic. |

The launcher coordinates exactly two nodes. It does not build or distribute images, synchronize
repository contents, or integrate with a cluster scheduler.
