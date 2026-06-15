#!/usr/bin/env python3
"""
run.py — MLPerf Training v6.0 (Llama 3.1 8B / FP4) launcher.

Single entry point for 1-node and 4-node runs. Generates per-node
launchers and dispatches them over SSH; each ssh stays foreground for
the life of the run, per-rank logs stream to LOGDIR, and run.py blocks
until every rank's ssh exits, returning the worst rc.

Examples
--------
  # List the configs we know about
  ./run.py --list-configs

  # 1-node smoke test on a1 (default host for 1-node)
  ./run.py --config MI350X_akash_1node

  # 4-node convergence run, random seed
  ./run.py --config MI350X_akash_4node

  # Pin a seed (for reproducibility / cache reuse on NFS)
  ./run.py --config MI350X_akash_4node --seed 32531

  # Override host list and pass extra env vars
  ./run.py --config MI350X_akash_4node \\
           --hosts mi350x-a1,mi350x-a2,mi350x-a3,mi350x-a4 \\
           --env NCCL_DEBUG=INFO --env PRIMUS_TRAIN_ITERS=200

  # Preview the per-node launcher without running anything
  ./run.py --config MI350X_akash_4node --dry-run

Detached / fire-and-forget
--------------------------
For runs that must survive an SSH disconnect, start run.py from inside
your own `screen`/`tmux` session, or wrap with nohup yourself. Example:

  ssh mi350x-a1
  cd ~/MLPerf-Training-v6.0-MiTAC/MiTAC/benchmarks/small_llm_pretraining/primus
  nohup python3 run.py --config MI350X_akash_4node --seed 32531 \\
        > /tmp/run.out 2>&1 < /dev/null &
  disown
  tail -F /tmp/run.out
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import shlex
import socket
import subprocess
import sys
import textwrap

# --------------------------------------------------------------------------
# Cluster + benchmark constants
# --------------------------------------------------------------------------

CONFIGS = {
    # --- MI325X cluster (mi325xr-[1-4]) -----------------------------------
    # 1-node configs have no `default_hosts` field — they always run on
    # whatever machine you invoke run.py on. `--hosts` is rejected.
    "MI325X_1node": {
        "config_file":     "config_MI325X_1node.sh",
        "wrapper":         "run_with_docker.sh",
        "container_name":  "mlperf_small_llm_pretraining",
        "nnodes":          1,
        "needs_master":    False,
        "needs_cache_dir": False,
        "fabric_subnet":   None,
    },
    "MI325X_4node": {
        "config_file":     "config_MI325X_4node.sh",
        "wrapper":         "run_with_docker_4node.sh",
        "container_name":  "mlperf_small_llm_pretraining_4node",
        "nnodes":          4,
        "default_hosts":   ["mi325xr-1", "mi325xr-2", "mi325xr-3", "mi325xr-4"],
        "needs_master":    True,
        "needs_cache_dir": False,  # TODO: enable when MI325X NFS path is wired
        "fabric_subnet":   None,   # TODO: set when MI325X cluster fabric IP range is known
    },
    # --- MI350X Akash rack (mi350x-a[1-4]) --------------------------------
    "MI350X_akash_1node": {
        "config_file":     "config_MI350X_akash_1node.sh",
        "wrapper":         "run_with_docker.sh",
        "container_name":  "mlperf_small_llm_pretraining",
        "nnodes":          1,
        "needs_master":    False,
        "needs_cache_dir": False,
        "fabric_subnet":   None,
    },
    "MI350X_akash_4node": {
        "config_file":     "config_MI350X_akash_4node.sh",
        "wrapper":         "run_with_docker_4node.sh",
        "container_name":  "mlperf_small_llm_pretraining_4node",
        "nnodes":          4,
        "default_hosts":   ["mi350x-a1", "mi350x-a2", "mi350x-a3", "mi350x-a4"],
        "needs_master":    True,
        "needs_cache_dir": True,
        "fabric_subnet":   "192.171.",
    },
    # --- MI355X (legacy 1-node slot) --------------------------------------
    "MI355X_1x8x1": {
        "config_file":     "config_MI355X_1x8x1.sh",
        "wrapper":         "run_with_docker.sh",
        "container_name":  "mlperf_small_llm_pretraining",
        "nnodes":          1,
        "needs_master":    False,
        "needs_cache_dir": False,
        "fabric_subnet":   None,
    },
}

DEFAULTS = {
    "datadir":     "/mlperf/llama3_1_8b/data",
    "modeldir":    "/mlperf/llama3_1_8b/tokenizer",
    "cache_dir":   "/mnt/shared/llama3_1_8b/npy_indices",
    "cont":        "rocm/amd-mlperf:llama3.1_8b_v6.0_MI350X",
    "master_addr": "192.171.3.12",
    "master_port": 29502,
    "primus_dir":  "~/MLPerf-Training-v6.0-MiTAC/MiTAC/benchmarks/small_llm_pretraining/primus",
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--config", choices=sorted(CONFIGS), help="benchmark config")
    p.add_argument("--list-configs", action="store_true", help="list available configs and exit")
    p.add_argument("--hosts",
                   help="comma-separated host list (multi-node configs only; "
                        "1-node configs always run locally and reject this flag)")
    p.add_argument("--master", default=DEFAULTS["master_addr"],
                   help=f"MASTER_ADDR for multi-node rendezvous (default: {DEFAULTS['master_addr']})")
    p.add_argument("--master-port", type=int, default=DEFAULTS["master_port"])
    p.add_argument("--seed", type=int,
                   help="SEED (default: random in [1,65535]); same value is sent to every rank")
    p.add_argument("--nexp", type=int, default=1, help="number of trials")
    p.add_argument("--datadir",   default=DEFAULTS["datadir"])
    p.add_argument("--modeldir",  default=DEFAULTS["modeldir"])
    p.add_argument("--cache-dir", default=DEFAULTS["cache_dir"],
                   help="shared NFS cache dir (multi-node only)")
    p.add_argument("--logdir",
                   help="LOGDIR on each node (default: ~/mlperf_logs/<config>_<ts>)")
    p.add_argument("--cont",       default=DEFAULTS["cont"], help="docker image tag")
    p.add_argument("--primus-dir", default=DEFAULTS["primus_dir"],
                   help="path to primus/ on each node")
    p.add_argument("--clear-caches", type=int, default=0, choices=(0, 1),
                   help="CLEAR_CACHES (1 needs NOPASSWD sudo; default 0)")
    p.add_argument("--power-monitor", type=int, default=1, choices=(0, 1),
                   help="POWER_MONITOR (1=spawn per-trial PSU sidecar writing "
                        "LOGDIR/power/result_<i>/node_<idx>.txt for the MLPerf "
                        "submission; 0=skip, e.g. dev runs). Default 1.")
    p.add_argument("--verbose", action="store_true",
                   help="MLPERF_VERBOSE_LOGS=1 (preserves rank stderr)")
    p.add_argument("--dry-run", action="store_true",
                   help="print launchers and dispatch commands without executing")
    p.add_argument("--env", action="append", default=[], metavar="VAR=VAL",
                   help="extra env var exported in each launcher (repeatable)")
    return p.parse_args()


# --------------------------------------------------------------------------
# Per-node launcher
# --------------------------------------------------------------------------

def get_git_rev() -> str:
    """Capture the orchestrator's repo state so it lands in every log banner.

    Returned shape: '<short-sha>[+dirty]' or 'unknown' if not a git checkout.
    Uses run.py's own directory so submodule state is what we report.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=here,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=here,
        ).stdout.strip()
        return f"{rev}{'+dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _banner_lines(args, rank: int, seed: int, logdir: str, ts: str,
                  git_rev: str, hosts: list) -> list:
    """Lines that go between '======== run.py params (begin) ========' and end.

    Mix of literal values (resolved here) and `${VAR}` references (expanded by
    bash AFTER `source ./config_*.sh` so we capture the post-source state).
    Each line is grep-friendly: `<dotted.key>: <value>`.
    """
    cfg = CONFIGS[args.config]
    cmdline = " ".join(shlex.quote(a) for a in sys.argv)

    # Static values (known at orchestration time)
    static = [
        f"runpy.config:        {args.config}",
        f"runpy.ts:            {ts}",
        f"runpy.rank:          {rank}",
        f"runpy.nnodes:        {cfg['nnodes']}",
        f"runpy.hosts:         {','.join(hosts)}",
        f"runpy.seed:          {seed}",
        f"runpy.nexp:          {args.nexp}",
        f"runpy.image:         {args.cont}",
        f"runpy.datadir:       {args.datadir}",
        f"runpy.modeldir:      {args.modeldir}",
        f"runpy.cache_dir:     {args.cache_dir if cfg['needs_cache_dir'] else '(n/a)'}",
        f"runpy.master:        {args.master}:{args.master_port}" if cfg["needs_master"] else "runpy.master:        (n/a)",
        f"runpy.clear_caches:  {args.clear_caches}",
        f"runpy.power_monitor: {args.power_monitor}",
        f"runpy.verbose:       {1 if args.verbose else 0}",
        f"runpy.git_rev:       {git_rev}",
        f"runpy.cmdline:       {cmdline}",
    ]
    # bash-resolved values (echoed AFTER source/NIC so they're final)
    keys = [
        ("config.DGXSYSTEM",            "DGXSYSTEM"),
        ("config.NNODES",               "NNODES"),
        ("config.GPUS_PER_NODE",        "GPUS_PER_NODE"),
        ("config.EXP",                  "EXP"),
        ("hp.PRIMUS_MICRO_BATCH_SIZE",  "PRIMUS_MICRO_BATCH_SIZE"),
        ("hp.PRIMUS_GLOBAL_BATCH_SIZE", "PRIMUS_GLOBAL_BATCH_SIZE"),
        ("hp.PRIMUS_LR",                "PRIMUS_LR"),
        ("hp.PRIMUS_MIN_LR",            "PRIMUS_MIN_LR"),
        ("hp.PRIMUS_LR_WARMUP_ITERS",   "PRIMUS_LR_WARMUP_ITERS"),
        ("hp.PRIMUS_LR_DECAY_ITERS",    "PRIMUS_LR_DECAY_ITERS"),
        ("hp.PRIMUS_TRAIN_ITERS",       "PRIMUS_TRAIN_ITERS"),
        ("hp.PRIMUS_EVAL_INTERVAL",     "PRIMUS_EVAL_INTERVAL"),
        ("nccl.NCCL_SOCKET_IFNAME",     "NCCL_SOCKET_IFNAME"),
        ("nccl.GLOO_SOCKET_IFNAME",     "GLOO_SOCKET_IFNAME"),
        ("nccl.NCCL_IB_HCA",            "NCCL_IB_HCA"),
        ("nccl.NCCL_IB_GID_INDEX",      "NCCL_IB_GID_INDEX"),
        ("nccl.NCCL_NET_GDR_LEVEL",     "NCCL_NET_GDR_LEVEL"),
        ("nccl.NCCL_ALGO",              "NCCL_ALGO"),
        ("nccl.NCCL_MIN_NCHANNELS",     "NCCL_MIN_NCHANNELS"),
    ]
    bash = [f'echo "{label}: ${{{var}:-(unset)}}"' for label, var in keys]
    extra_bash = [f'echo "extra.{kv.split("=", 1)[0]}: ${{{kv.split("=", 1)[0]}:-(unset)}}"'
                  for kv in args.env]
    return static, bash + extra_bash


def build_launcher(args, rank: int, seed: int, logdir: str, ts: str,
                   git_rev: str, hosts: list) -> str:
    """Render the bash launcher that runs on the target node."""
    cfg = CONFIGS[args.config]

    extras = "\n".join(f"export {kv}" for kv in args.env)
    cache_export = f'export CACHE_DIR="{args.cache_dir}"' if cfg["needs_cache_dir"] else ""
    master_exports = (
        f'export MASTER_ADDR="{args.master}"\n'
        f'export MASTER_PORT={args.master_port}\n'
        f'export NODE_RANK={rank}'
    ) if cfg["needs_master"] else ""
    verbose_export = "export MLPERF_VERBOSE_LOGS=1" if args.verbose else ""

    nic_block = ""
    if cfg["nnodes"] > 1 and cfg.get("fabric_subnet"):
        # awk-friendly form: "192[.]171[.]" matches "192.171." as a literal.
        subnet_re = cfg["fabric_subnet"].replace(".", "[.]")
        nic_block = textwrap.dedent(f"""\
            # Pick the cluster fabric NIC: an interface with an IP under the
            # configured subnet ({cfg['fabric_subnet']}*). Necessary because
            # mismatched NIC names (e.g. ens512np0 vs ens513np0 on a3) make
            # Gloo bind to lo and fail to form the rendezvous mesh.
            NIC=$(ip -o -4 addr show | awk '$4 ~ /^{subnet_re}/ {{print $2; exit}}')
            if [[ -n "$NIC" ]]; then
              export GLOO_SOCKET_IFNAME=$NIC
              export NCCL_SOCKET_IFNAME=$NIC
            fi""")

    static_lines, bash_echos = _banner_lines(args, rank, seed, logdir, ts, git_rev, hosts)
    static_block = "\n".join(static_lines)
    banner_block = (
        "# === run.py parameter banner (machine-parseable; grep '^==RUNPY' to extract) ===\n"
        'echo "==RUNPY-PARAMS-BEGIN=="\n'
        'echo "runpy.host:          $(hostname)"\n'
        + "\n".join(f'echo {shlex.quote(line)}' for line in static_lines)
        + "\n"
        + "\n".join(bash_echos)
        + "\necho \"==RUNPY-PARAMS-END==\""
    )

    parts = [
        "#!/bin/bash",
        "set -e",
        f'cd {args.primus_dir}',
        f'export CONT="{args.cont}"',
        f'export DATADIR="{args.datadir}"',
        f'export MODELDIR="{args.modeldir}"',
        f'export LOGDIR="{logdir}"',
        f'export CONFIG_NAME="{cfg["config_file"]}"',
        f'export CLEAR_CACHES={args.clear_caches}',
        f'export POWER_MONITOR={args.power_monitor}',
        f'export NEXP={args.nexp}',
        f'export SEED={seed}',
        cache_export,
        master_exports,
        verbose_export,
        extras,
        f'source ./{cfg["config_file"]}',
        nic_block,
        banner_block,
        f'exec bash {cfg["wrapper"]}',
    ]
    return "\n".join(p for p in parts if p.strip()) + "\n"


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def dispatch_node(host: str, rank: int, launcher: str, logdir: str, ts: str,
                  dry_run: bool):
    """Start the rank's launcher on `host` under a foreground ssh.

    The ssh stays connected for the entire training run; per-rank stdout
    and stderr stream back and are written to a per-rank log file. The
    ssh exits when training exits, and the orchestrator does Popen.wait()
    to detect completion and propagate rc.

    Returns the Popen handle, or None if dry_run.
    """
    log_file = f"{logdir}/{host}_rank{rank}.out"
    script_path = f"/tmp/run_py_{ts}_rank{rank}.sh"

    if dry_run:
        print(f"--- [DRY-RUN] {host} rank={rank} ---")
        print(textwrap.indent(launcher, "  "))
        print(f"  → log: {log_file}")
        return None

    os.makedirs(logdir, exist_ok=True)
    log_fp = open(log_file, "wb")

    # If the target host is the orchestrator itself, skip ssh and just run
    # bash directly. Avoids needing ssh-to-self to be configured (which is
    # not the default on every node) and removes a layer of indirection.
    if _is_local_host(host):
        with open(script_path, "w") as f:
            f.write(launcher)
        os.chmod(script_path, 0o755)
        print(f"[run.py] starting {host} (rank {rank}, local); streaming → {log_file}")
        return subprocess.Popen(
            ["bash", script_path],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )

    remote_cmd = (
        f"mkdir -p {shlex.quote(logdir)} && "
        f"cat > {script_path} <<'__RUNPY_EOF__'\n"
        f"{launcher}"
        f"__RUNPY_EOF__\n"
        f"chmod +x {script_path} && exec bash {script_path}"
    )
    print(f"[run.py] starting {host} (rank {rank}); streaming → {log_file}")
    return subprocess.Popen(
        ["ssh", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
         host, remote_cmd],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )


def wait_for_all(procs: list, hosts: list) -> int:
    """Wait on all per-rank Popens and return the worst rc.

    Each ssh stays connected for the duration of training and exits when
    its rank exits. We block until all of them are done. If any rank
    returned non-zero we surface that so the caller (e.g. sweep.sh) marks
    the run failed.
    """
    rcs = {}
    print("[run.py] waiting on all ranks (Ctrl+C interrupts; child sshs survive)")
    try:
        for proc, host in zip(procs, hosts):
            rc = proc.wait()
            rcs[host] = rc
            print(f"[run.py] {host} exited rc={rc}")
    except KeyboardInterrupt:
        print("[run.py] interrupted; leaving sshs running")
        return 130
    worst = max(rcs.values())
    if worst != 0:
        bad = [h for h, rc in rcs.items() if rc != 0]
        print(f"[run.py] FAILED on: {', '.join(bad)} (rcs={rcs})")
    return worst


def _local_aliases() -> set:
    """Names that all refer to the current machine."""
    aliases = {"localhost", "127.0.0.1"}
    try:
        h = socket.gethostname()
        aliases.add(h)
        aliases.add(h.split(".")[0])  # short name
        aliases.add(socket.getfqdn())
    except Exception:
        pass
    return aliases


def _is_local_host(host: str) -> bool:
    """True if `host` resolves to (or equals) the orchestrator's own machine."""
    return host in _local_aliases()


def preflight_ssh(hosts: list[str]) -> None:
    """Verify each remote host is reachable. Skip local hosts (no ssh needed)."""
    failed = []
    for h in hosts:
        if _is_local_host(h):
            continue
        rc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", h, "true"],
            capture_output=True,
        ).returncode
        if rc != 0:
            failed.append(h)
    if failed:
        sys.exit(f"error: SSH unreachable: {', '.join(failed)}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.list_configs:
        for name, cfg in sorted(CONFIGS.items()):
            if cfg["nnodes"] == 1:
                hosts_str = "(local)"
            else:
                hosts_str = ",".join(cfg.get("default_hosts", []))
            print(f"{name:30s}  nnodes={cfg['nnodes']}  hosts={hosts_str}")
        return 0

    if not args.config:
        sys.exit("error: --config is required (use --list-configs to see options)")

    cfg = CONFIGS[args.config]
    if cfg["nnodes"] == 1:
        # 1-node: always runs on the orchestrator's machine. No SSH, no
        # --hosts override — the host is implicitly the local one.
        if args.hosts:
            sys.exit(
                f"error: --hosts is not allowed for 1-node config "
                f"'{args.config}' (it always runs locally)"
            )
        hosts = [socket.gethostname()]
    else:
        hosts = args.hosts.split(",") if args.hosts else cfg["default_hosts"]
        if len(hosts) != cfg["nnodes"]:
            sys.exit(
                f"error: --config {args.config} expects {cfg['nnodes']} host(s), "
                f"got {len(hosts)}: {hosts}"
            )

    seed = args.seed if args.seed is not None else random.randint(1, 65535)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = args.logdir or f"/home/mitac/mlperf_logs/{args.config}_{ts}"
    git_rev = get_git_rev()

    print(f"[run.py] config:    {args.config}")
    print(f"[run.py] hosts:     {hosts}")
    print(f"[run.py] seed:      {seed}{' (specified)' if args.seed is not None else ' (random)'}")
    print(f"[run.py] logdir:    {logdir}")
    print(f"[run.py] image:     {args.cont}")
    print(f"[run.py] git_rev:   {git_rev}")
    if cfg["needs_master"]:
        print(f"[run.py] master:    {args.master}:{args.master_port}")
    if cfg["needs_cache_dir"]:
        print(f"[run.py] cache:     {args.cache_dir}")
    print()

    if not args.dry_run:
        preflight_ssh(hosts)

    procs = []
    for rank, host in enumerate(hosts):
        launcher = build_launcher(args, rank, seed, logdir, ts, git_rev, hosts)
        proc = dispatch_node(host, rank, launcher, logdir, ts, args.dry_run)
        if proc is not None:
            procs.append(proc)

    if args.dry_run:
        return 0

    print(f"[run.py] all {len(procs)} ranks running. Live logs:")
    for rank, host in enumerate(hosts):
        print(f"          tail -F {logdir}/{host}_rank{rank}.out")
    print()
    return wait_for_all(procs, hosts)


if __name__ == "__main__":
    sys.exit(main())
