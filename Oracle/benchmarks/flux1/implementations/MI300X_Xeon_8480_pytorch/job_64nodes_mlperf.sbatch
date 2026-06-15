#!/bin/bash
#SBATCH --partition=mlperf
#SBATCH --gres=gpu:8
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=flux1_64nodes
#SBATCH --output=flux1_64nodes_%j.out
#SBATCH --error=flux1_64nodes__%j.err
#SBATCH --exclusive
#SBATCH --comment="mlperf"

set -euxo pipefail

export REPO_DIR=${PWD}
export CONFIG_DIR=${PWD}

cd "${REPO_DIR}"

export SCHNELL_YAML="${REPO_DIR}/conf/model/schnell.yaml"
grep -nE 'enable_cuda_graph|cuda_graph_impl|cuda_graph_scope|cpu_offloading' "${SCHNELL_YAML}"

export DGXSYSTEM=MI355X_64x08x16
export BATCHSIZE=4096
export LEARNING_RATE=${LEARNING_RATE:-"0.0004"}
export MBS=8
export DATADIR=/mnt/localdisk/flux1_dataset
export LOGDIR=.
export CONT=rocm/amd-mlperf:flux1_training_6.0 #Your image name
export CONT_TAR=${CONT_TAR:-/mnt/localdisk/container-images/amd-mlperf_flux1_training_6.0.tar} #path to container tar if not in registry

export LOCAL_DATADIR=/mnt/localdisk/flux1_dataset #path to dataset
export ROOT_SEED=$RANDOM
export RUNTIME_TUNABLES=0
export STAGE_TO_LOCAL_NVME=0
export STAGE_FANOUT=0
export LOCAL_DATADIR_BASE=${LOCAL_DATADIR_BASE:-/mnt/localdisk}
export LOCAL_STAGE_WORKERS=${LOCAL_STAGE_WORKERS:-64}
export FORCE_DOCKER_LOAD=${FORCE_DOCKER_LOAD:-0}
export ULIMIT_NOFILE=${ULIMIT_NOFILE:-1048576}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-/results/nccl_%h_%p.log}
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-INFO}
export EXTRA_ARGS="${EXTRA_ARGS:-model.flux_params.flux_config.enable_cuda_graph=False model.flux_params.flux_config.cuda_graph_scope=full_iteration model.flux_params.flux_config.cpu_offloading=True model.flux_params.flux_config.cpu_offloading_activations=True model.flux_params.flux_config.cpu_offloading_weights=False}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  bash -lc '
    if [[ "${FORCE_DOCKER_LOAD}" == "1" ]]; then
      sudo docker load -i "$CONT_TAR"
    elif sudo docker image inspect "$CONT" >/dev/null 2>&1; then
      echo "Image already present on $(hostname): $CONT"
    else
      sudo docker load -i "$CONT_TAR"
    fi
  '

srun bash "${REPO_DIR}/run_with_docker_slurm.sh"
