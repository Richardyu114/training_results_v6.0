export DATADIR=/home/scratch.rachitg_gpu_1/mlperf-t/gov_report
export MODEL=/home/scratch.rachitg_gpu_1/mlperf-t/model
export UCX_TLS=self,tcp

# JET
export JET_DIR="/project/coreai_mlperf_training/common/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh"

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized/-/blob/main"

############# Cosmetic stuff for submission system.json files ###############
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="onprem"
export MLPERF_SYSTEM_NAME="NVIDIA H200 (8x H200-SXM-141GB-CTS)"
export MLPERF_SYSJSON_SYSNAME_INCLUDE_NUM_NODES=0
export MLPERF_FRAMEWORK="NVIDIA NeMo Framework Release 24.09"
export MLPERF_FRAMEWORK_SHORT_NAME="ngc24.09_nemo"

export MLPERF_HOST_STORAGE_TYPE="XXX"
export MLPERF_HOST_STORAGE_CAPACITY="XXX"
export MLPERF_HOST_NETWORKING="XXX"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="XXX"
export MLPERF_ACCELERATOR_INTERCONNECT="XXX"
