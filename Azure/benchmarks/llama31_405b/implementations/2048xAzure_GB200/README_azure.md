# Azure MRC Workflow for LLaMA 3.1 405B (v6.0)

## Config Organization

The `configs/` directory is split into two categories:

- **`nvidia/`** — Original NVIDIA configs from the container
- **`azure/`** — Azure MRC-specific configs that source the NVIDIA base configs 

## Launch Workflow

```bash
python launch_405b.py --hostfile /path/to/hostfile --minibs 144
python launch_405b.py --hostfile /path/to/hostfile --minibs 144 --time 12:00:00
```

The python launcher sources `run_sweep.sh` automatically, computes `DP = nodes / 16`, and runs `sbatch` from the current directory
