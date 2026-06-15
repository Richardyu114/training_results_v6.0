#!/usr/bin/env python3
"""Launch a 405B v6.0 experiment from a hostfile and mini-batch size.

Usage:
    python launch_405b.py --hostfile /path/to/hostfile --minibs 144
    python launch_405b.py --hostfile /path/to/hostfile --minibs 144 --time 12:00:00
    python launch_405b.py --hostfile /path/to/hostfile --minibs 144 --profile
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The v6.0 tree IS the implementation directory (run.sub lives here)
V6_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(REPO_ROOT, "profiling")


def count_nodes(hostfile: str) -> int:
    with open(hostfile) as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch 405B experiment (v6.0)")
    parser.add_argument("--hostfile", required=True, help="Path to hostfile (one host per line)")
    parser.add_argument("--minibs", required=True, type=int, help="Mini-batch size per DP rank")
    parser.add_argument("--time", default="24:00:00", help="Job time limit (default: 24:00:00)")
    parser.add_argument("--config", default=None,
                        help="Config file to source (default: auto-select via run_sweep.sh)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable nsys profiling (sources profiling config and swaps run_and_time.sh)")
    args = parser.parse_args()

    hostfile = os.path.abspath(args.hostfile)
    if not os.path.isfile(hostfile):
        sys.exit(f"ERROR: hostfile not found: {hostfile}")

    num_nodes = count_nodes(hostfile)
    if num_nodes == 0:
        sys.exit("ERROR: hostfile is empty")
    if num_nodes % 16 != 0:
        sys.exit(f"ERROR: number of nodes ({num_nodes}) must be divisible by 16")

    dp = num_nodes // 16

    if args.config:
        config_cmd = f"source {os.path.abspath(args.config)}"
    else:
        config_cmd = f"source {V6_DIR}/configs/azure/run_sweep.sh {dp} {args.minibs}"

    if args.profile:
        profile_config = os.path.join(PROFILE_DIR, "config_performance_profile.sh")
        config_cmd += f"\nsource {profile_config}"

    # Build a bash script that sources the config, cd's to v6.0, and sbatch's
    script = f"""\
set -euo pipefail
{config_cmd}
echo "LOGDIR=${{LOGDIR}}"
cd {V6_DIR}
sbatch --nodelist=$(paste -sd, {hostfile}) --time={args.time} run.sub
"""

    mrc_mounts = os.environ.get("MRC_MOUNTS_FILE", "(default: infra/mrc_mounts.sh)")
    mode = "PROFILE" if args.profile else "TRAINING"
    print(f"Mode: {mode}")
    print(f"Nodes: {num_nodes}  DP: {dp}  MINIBS: {args.minibs}  TIME: {args.time}")
    print(f"MRC mounts: {mrc_mounts}")
    print()

    ret = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    print(ret.stdout, end="")
    if ret.stderr:
        print(ret.stderr, end="", file=sys.stderr)
    # Print LOGDIR and job info prominently
    for line in ret.stdout.splitlines():
        if line.startswith("LOGDIR="):
            print(f"\n{line}")
        if "Submitted batch job" in line:
            job_id = line.strip().split()[-1]
            print(f"Job ID: {job_id}")
            print(f"Monitor: squeue -j {job_id}")
            print(f"Logs:    tail -f slurm-{job_id}.out")
            break
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
