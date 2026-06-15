# Traceback dumping scripts

## Scripts
1. [dump_tracebacks_node.sh](dump_tracebacks_node.sh) - when run inside the container, dumps `py-spy`, `gdb` and `cuda-gdb` tracebacks for all processes listed in `nvidia-smi`
2. [hang_monitor.sh](hang_monitor.sh) - provides functions to start a background process that launches given commands a few minutes before the slurm timout.
3. [traceback_analysis.py](traceback_analysis.py) - run on the traceback folder to gather the top few lines of the stack trace from all ranks.
4. [traceback_analysis.sh](traceback_analysis.sh) - bash version of traceback_analysis.py. Suitable for systems without python installed

## Usage
In run.sub `hang_monitor` is set up to launch an overlapping srun job `HANG_MONITOR_TIMEOUT` minutes before the slurm timeout.
The srun job starts one `dump_tracebacks_node.sh` script instance on every node.
By default, `HANG_MONITOR_TIMEOUT` is set to 0 for jobs shorter than 1h (based on WALLTIME variable) and to 5 otherwise.

## Results
The output of the `hang_monitor` and `dump_tracebacks_node.sh` script is saved to `${LOGDIR}/<prefix>_hang_monitor.log` log file.
The tracebacks are dumped to `${LOGDIR}/tracebacks/<prefix>/rank<rank>_pid<pid>.<ext>` where:
- `<rank>` is the global rank of the given process (SLURM_PROCID set for that process by Slurm)
- `<pid>` is the process id the given process (unique within a node)
- `<ext>` is `pyspy`, `gdb` or `cuda-gdb` depending on the tracing program

## Known limitations
1. `cuda-gdb` is currently known to crash the training.
This won't matter for a hung job, but to make sure pyspy and gdb traceback are collected properly,
cuda-gdb dumping is done in a separate loop over processes.

## Examples of analyzing the tracebacks
1. Check active Python functions: `grep -A 1 -e "(active" *.pyspy`
2. Run `traceback_analysis.py` on the traceback folder to gather the top few lines of the stack trace from all ranks.
    - The script also has an `--aggregate` flag that groups by stack trace rather than rank, so that we can easily observe
      which ranks have a common stack trace.
    - It looks for a main thread to extract the stack trace from, whose thread ID may need to be supplied through `--main-thread-tid=[TID]`.
      The main thread ID can be found in the pyspy log, as in `Thread 0x1555551A6740 (active): "MainThread"`.
    - The maximum number of lines to be extracted from the pyspy/GDB logs can be set with `--pyspy-max-lines=[int]` and `--gdb-max-lines=[int]`.
    - The output file to be written out can be specified with `--output=[path to output file]`.
    - Example usage: `python3 traceback_analysis.py --aggregate --main-thread-tid=0x1555551a6740 --pyspy-max-lines=15 --gdb-max-lines=15 --output=/home/jaeminc/traceback_analysis.md ./traceback_folder`
    - Example output (with `--aggregate`, which is on by default):
      ```layernorm_fwd_fp8...```
      0,1,2,3,8,9,10,11
    - Example output (with `--no-aggregate`):
      |     | pyspy                      | GDB                                               |
      |----:|:--------------------------:|:--------------------------------------------------|
      |   0 | ```layernorm_fwd_fp8...``` | ```#0  0x00001555552b3c9b in sched_yield ()...``` |

3. Running `traceback_analysis.sh` results in the same output as `traceback_analysis.py`. Usage:
    - bash traceback_analysis.sh <traceback_dir> [<ouput_file> <aggregate> <main-thread-tid> <pyspy-max-lines> <gdb-max-lines]
