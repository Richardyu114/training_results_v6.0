#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
set -e

###########################################################################
# This script is invoked inside the container, and a copy is launched on every
# rank.
#
# This script MUST NOT have any SLURM dependences (no use of SLURM envvars)
#
# If this is a pytorch benchmark this script assumes that it is invoked with
# something (torchrun, srun+enroot, srun+slurm2pytorch, or
# mpirun+slurm2pytorch) that correctly sets the variables described at
# https://pytorch.org/docs/stable/elastic/run.html#environment-variables RANK,
# LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT
#
# To run this script interactively you must invoke it with torchrun
#
# if you need NODE_RANK you can derive it by
# NODE_RANK=$(( RANK / LOCAL_WORLD_SIZE ))
#
# If this script is for a framework that assumes mpirun (or srun --mpi), the
# envvars the envvars described at
# https://docs.open-mpi.org/en/v5.0.x/tuning-apps/environment-var.html
# OMPI_COMM_WORLD_SIZE, OMPI_COMM_WORLD_RANK, OMPI_COMM_WORLD_LOCAL_SIZE,
# OMPI_COMM_WORLD_LOCAL_RANK, OMPI_COMM_WORLD_NODE_RANK
###########################################################################

# vars that should be set by the launcher (Pytorch)
: "${RANK:?RANK not set}"
: "${LOCAL_RANK:?LOCAL_RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${LOCAL_WORLD_SIZE:?LOCAL_WORLD_SIZE not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"

[ "${DEBUG}" = "0" ] && set -x

# Vars without defaults
: "${SEED:?SEED not set}"
: "${WALLTIME:=?WALLTIME not set}"

: "${NVTX_FLAG:=0}"
: "${COMM_GEMM_OVERLAP_TEST:=0}"
: "${COMM_GEMM_OVERLAP_TEST_CLEANUP:=1}"

# Get rank to hostname mapping
if [ "$LOCAL_RANK" -eq 0 ]; then
  echo "Hello from: $(hostname)"
fi

export NVTE_NORM_FWD_USE_CUDNN=0
export NVTE_NORM_BWD_USE_CUDNN=0
cp /workspace/transformerengine/tests/pytorch/distributed/run_layer_with_overlap.py .
sleep 5

declare -a CMD
#if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
if [[ ${LOCAL_WORLD_SIZE} -gt 1 ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( 'python' '-u')
else
    # interactive run on single node, no need to bind
    CMD=( 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi
if [ ${COMM_GEMM_OVERLAP_TEST} -gt 0 ]; then
  # run benchmark
  [[ "$RANK" -eq 0 ]] && echo "Running COMM-GEMM overlap test"

  layers=("qkv" "proj" "fc1" "fc2")
  for layer in "${layers[@]}"
  do
    [[ "$RANK" -eq 0 ]] && echo "Running ${layer}"
    NSYSCMD=" LAYER=${layer} nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --output /results/${layer}_${SLURMD_NODENAME}_${SLURM_NODEID}_r${SLURM_PROCID}.nsys-rep ${CMD[@]} benchmark_comm_gemm_overlap.py "
    [[ "$RANK" -eq 0 ]] && echo ${NSYSCMD}
    eval "${NSYSCMD}"
    NSYS_JSON_CMD=" nsys stats -r cuda_gpu_trace -f json -o /results/${layer}_${SLURMD_NODENAME}_${SLURM_NODEID}_r${SLURM_PROCID} --force-export=true /results/${layer}_${SLURMD_NODENAME}_${SLURM_NODEID}_r${SLURM_PROCID}.nsys-rep"
    [[ "$RANK" -eq 0 ]] && echo ${NSYS_JSON_CMD}
    ${NSYS_JSON_CMD}
  done
fi

[[ "$RANK" -eq 0 ]] && echo "Parsing nsys reports!"
LAYER=all ACTION=parse CUDA_GPU_TRACE_DIR=/results ${CMD[@]} benchmark_comm_gemm_overlap.py |& tee "/results/${SPREFIX}_comm_gemm_overlap.log"

if [ ${COMM_GEMM_OVERLAP_TEST_CLEANUP} -gt 0 ] && [[ "$RANK" -eq 0 ]]; then
  echo "Cleaning up!"
  rm /results/*.nsys-rep
  rm /results/*.sqlite
  rm /results/qkv*.json /results/proj*.json /results/fc1*.json /results/fc2*.json
fi

set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

if [[ "$RANK" -eq 0 ]]; then
  # End timing
  END=$(date +%s)
  END_FMT=$(date +%Y-%m-%d\ %r)
  echo "ENDING TIMING RUN AT ${END_FMT}"

  # Report result
  RESULT=$(( END - START ))
  RESULT_NAME="large language model"
  echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"
fi
