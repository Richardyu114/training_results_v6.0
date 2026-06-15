## data path
export DATADIR=/raid/michalm/data/text/govreport/mlperf
export MODEL=/raid/michalm/data/model/llama2_70b_pyt/nemo_mode-pretrain/mlperf_bf16
export SLURM_MPI_TYPE=pmi2
export CLEAR_CACHES=0
export MELLANOX_VISIBLE_DEVICES=all
export UCX_TLS=self,tcp
export ENABLE_IB_BINDING=0

# JET
export JET_DIR="/raid/michalm/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh "

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized/-/blob/main"

############# Cosmetic stuff for submission system.json files ###############
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="Preview"
export MLPERF_SYSTEM_NAME="dgx_b200_preview"
export MLPERF_SYSJSON_SYSNAME_INCLUDE_NUM_NODES=1
export MLPERF_FRAMEWORK="NVIDIA NEMO"
export MLPERF_FRAMEWORK_SHORT_NAME="nemo_preview"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY=""
export MLPERF_HOST_NETWORKING=""
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA HGX B200"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION=""
export MLPERF_ACCELERATOR_INTERCONNECT=""
