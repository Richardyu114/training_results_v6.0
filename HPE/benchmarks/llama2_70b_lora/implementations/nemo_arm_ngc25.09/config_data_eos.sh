export DATADIR="/lustre/fsw/coreai_mlperf_training/data/lora/gov_report"
export MODEL="/lustre/fsw/coreai_mlperf_training/data/lora/model"
export UCX_TLS=self,tcp

# JET
export JET_DIR="/project/coreai_mlperf_training/common/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh"

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized/-/blob/main"

############# Cosmetic stuff for submission system.json files ###############
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="Available on-premise"
export MLPERF_SYSTEM_NAME="Eos"
export MLPERF_SYSJSON_SYSNAME_INCLUDE_NUM_NODES=1
export MLPERF_FRAMEWORK="NVIDIA NeMo Framework Release 24.09"
export MLPERF_FRAMEWORK_SHORT_NAME="ngc24.09_nemo"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="2x 1.92TB NVMe SSD + 8x 3.84TB NVMe U.2"
export MLPERF_HOST_NETWORKING="Storage: 2x ConnectX-7 IB NDR 400Gb/s, Compute: 8x ConnectX-7 IB NDR 400Gb/s, Management: 100Gb/s Ethernet NIC"
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA H100-SXM5-80GB"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="HBM3"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLINK Gen4 900 GB/s + NVSWITCH Gen3"
