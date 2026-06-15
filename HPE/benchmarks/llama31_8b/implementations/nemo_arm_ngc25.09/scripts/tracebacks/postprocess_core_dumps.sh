#!/bin/bash

# This script postprocesses existing CUDA core dumps with cuda-gdb.
# The only requirement is `cuda-gdb` program (doesn't even need GPU).
#
# Expecting the following env variables to be set externally:
#   - CUDA_COREDUMP_BASEDIR - directory with core dumps
#   - REMOVE_CUDA_GDB_CORE_DUMP - if 1, removes coredump after successful postprocessing
#   - CUDA_GDB_BIN - path to binary which executes cuda-gdb dump

set -x

: "${REMOVE_CUDA_GDB_CORE_DUMP:=1}"
: "${CUDA_GDB_BIN:=cuda-gdb}"

echo "$0: Processing CUDA cores in ${CUDA_COREDUMP_BASEDIR}"

for dump in ${CUDA_COREDUMP_BASEDIR}/*.nvcudmp; do

  savefile="${dump%.nvcudmp}.cuda-gdb"
  ${CUDA_GDB_BIN} -batch -ex "target cudacore $dump" -ex 'info cuda kernels' -ex 'info cuda devices' >> "$savefile"

  if [ $? -eq 0 ] && [ "${REMOVE_CUDA_GDB_CORE_DUMP}" == "1" ] ; then
    echo "Removing core dump $dump"
    rm $dump
  fi
done

echo "Postprocessing CUDA cores done on host $(hostname)"
