#!/bin/bash

# This script should be called with on a compute node *inside* a container.
# Steps:
# 1. Installs py-spy and gdb
# 2. Finds all processes listed in nvidia-smi
# 3. Dumps traceback for all of them to a subdirectory of ${RESULTS_DIR}/${TRACEBACKS_ID}
#
# Expecting the following env variables to be set externally:
#   - TRACEBACKS_ID - for differentiating dump sessions
#   - RESULTS_DIR - for storing the tracebacks
#   - ATTEMPT_CUDA_GDB_DUMP - if 1, the script will attempt to dump tracebacks with cuda-gdb.
#                             Default is 0 since cuda-gdb might break the application and doesn't work on preos.
#   - CUDA_GDB_BIN - path to binary which executes cuda-gdb dump

set -x

: "${TRACEBACKS_ID:=''}"
: "${RESULTS_DIR:=/results/tracebacks}"
: "${ATTEMPT_CUDA_GDB_TRACEBACKS_DUMP:=0}"
: "${CUDA_GDB_BIN:=cuda-gdb}"

RESULTS_PATH=${RESULTS_DIR}/${TRACEBACKS_ID}
echo "$0: dumping tracebacks to ${RESULTS_PATH}"
mkdir -p "${RESULTS_PATH}"

# Install required tools
(
  pip install py-spy
  export DEBIAN_FRONTEND=noninteractive;
  apt update
  apt install -y gdb
) &> /dev/null

# Find processes

# This queries for PIDs listed in nvidia-smi output (see https://stackoverflow.com/a/56181444)
pids=$(nvidia-smi -q -x | grep pid | sed -e "s/<pid>//g" -e "s/<\/pid>//g" -e "s/^[[:space:]]*//" | sort -u)
# Alternative (queries PIDs of all python processes, which might be too much in some cases):
#pids='$(ps -C "python -u /workspace/llm" -o pid=)'

echo "Matching processes: $pids"

# First dump pyspy and gdb since they work reliably
for pid in ${pids}; do
  if ps -p $pid > /dev/null; then

    slurm_procid=`cat /proc/$pid/environ | tr "\0" "\n" | grep SLURM_PROCID`
    slurm_procid=${slurm_procid:13}
    printf -v slurm_procid "%05d" $slurm_procid

    savefile="$RESULTS_PATH/rank${slurm_procid}_pid${pid}.pyspy"
    hostname &> "$savefile"
    date +'%y%m%d%H%M%S%N' &>> "$savefile"
    py-spy dump --pid $pid &>> "$savefile"

    savefile="$RESULTS_PATH/rank${slurm_procid}_pid${pid}.gdb"
    hostname &> "$savefile"
    date +'%y%m%d%H%M%S%N' &>> "$savefile"
    gdb -p $pid -batch -ex 'thread apply all bt' &>> "$savefile"

  else
    echo "Process $pid doesnt exist anymore"
  fi
done

if [ "${ATTEMPT_CUDA_GDB_TRACEBACKS_DUMP}" -eq 1 ]; then

  sleep 30  # let other nodes finish dumping

  # Second loop over processes dumps cuda-gdb. This might crash the application
  for pid in ${pids}; do
    if ps -p $pid > /dev/null; then

      slurm_procid=`cat /proc/$pid/environ | tr "\0" "\n" | grep SLURM_PROCID`
      slurm_procid=${slurm_procid:13}
      printf -v slurm_procid "%05d" $slurm_procid

      savefile="$RESULTS_PATH/rank${slurm_procid}_pid${pid}.cuda-gdb"
      hostname &> "$savefile"
      date +'%y%m%d%H%M%S%N' &>> "$savefile"
      ${CUDA_GDB_BIN} -batch -ex 'info cuda kernels' -ex 'info cuda devices' -p $pid &>> "$savefile"

    else
      echo "Process $pid doesnt exist anymore"
    fi
  done
fi

echo "Dumping done on host $(hostname)"
