# Single-node preflights on GB200

## Launch jobs

`getnodelist` is a script that uses `sinfo` to find all the nodes that are
currently "alive" in the cluster.

First I run a single node run on every node

```
# In optimized/llama31_405b_llm/pytorch
for i in $(/mnt/home/jaeminc/optimized/llama31_405b_llm/pytorch/scripts/cw-ibm-backups/getnodelist); do
    launch-mlperf-benchmark.sh --partition=gb200 \
    --container /mnt/vast/proj/nvidia/mlperft/containers/nvidian+gpuhw+mlperft+llama31_405b_llm.pytorch.27564994.sqsh \
    --config config_GB200_1x4x34xtp2pp2cp1.sh  \
    --slurm-extra=--output=/mnt/vast/proj/nvidia/mlperft/raw_logs/single-node-preflights/single-node-preflight-250428-1018/%N-%j.out \
    --slurm-extra=--nodelist=$i
done
```

Then wait for all the jobs to get to `run_stop`.  Note that in the current
container with cuda graphs the jobs don't exit after `run_stop`, so you need to
watch and then `scancel` all the jobs that are finished.  (Or set the WALLTIME
more carefully than I have.)

## Analyze the logs

```
# In preflights result folder
for i in *.out; do
    echo -e "$i\t$(/mnt/home/jaeminc/optimized/mlperf_utils/mllog-converge.py $i)"
done | cut -f1,4 | sed -E -e 's/-.....out//' | tee sweep-summary-$(date +"%y%m%d-%H%M")
```

The `sed -E -e 's/-.....out//' is stripping the job number and suffix off the
file name (which are of form Node-name-jobnum.out).

Sweep-summary contains lines of the form `<Node-name <tab> score (in seconds)>`

## Print statistics and obtain outliers

Print stats and find outliers with 2% or more slowdown than the median:
```
python3 /mnt/home/jaeminc/optimized/llama31_405b_llm/pytorch/scripts/cw-ibm-backups/preflight-stats.py sweep-summary-250428-1757 1.02
```

This would print something like the following:
```
Min:    862.001
Avg:    881.452
Median: 881.170
Max:    907.166
Stdev:  7.575

Outlier condition: times > 1.02x median (898.793)

Outliers:
slurm-gb200-213-215     902.586 (1.02x median)
slurm-gb200-215-193     900.332 (1.02x median)
slurm-gb200-217-249     899.848 (1.02x median)
slurm-gb200-220-215     907.166 (1.03x median)
slurm-gb200-221-029     899.137 (1.02x median)
slurm-gb200-223-011     903.466 (1.03x median)
```
