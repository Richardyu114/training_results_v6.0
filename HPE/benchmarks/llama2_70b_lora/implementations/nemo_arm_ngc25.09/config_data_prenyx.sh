## data path
export DATADIR="/lustre/fsw/coreai_mlperf_training/data/lora/gov_report"
export MODEL="/lustre/fsw/coreai_mlperf_training/data/lora/model"
export SLURM_MPI_TYPE=pmix
export UCX_TLS=self,tcp
export UCX_NET_DEVICES=ens6f1np1
export GPU_ARCH="b"

# DLSIM
export DLSIM_DEVICE="Tesla_B200_SXM_180GB"
export DLSIM_METHODOLOGY="blackwell.optimistic"

# JET
export JET_DIR="/project/coreai_mlperf_training/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh "

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized/-/blob/main"

# Specify the name of the metrics file to use when capturing metrics with NSYS
export NSYS_METRICS_SET="gb10x"

############# Cosmetic stuff for submission system.json files ###############
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="Available on-premise"
export MLPERF_SYSTEM_NAME='Nyx (@MLPERF_SYS_SIZE@x NVIDIA DGX B200)'
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA Blackwell GPU (B200-SXM-180GB)"
export MLPERF_FRAMEWORK="NVIDIA NeMo Framework Release 25.04"
export MLPERF_FRAMEWORK_SHORT_NAME="ngc25.04_nemo"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="8x 3.5TB NVMe E1.s"
export MLPERF_HOST_NETWORKING="8x ConnectX-7 IB NDR 400Gb/s"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="HBM3e"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLINK Gen5 1800 GB/s + NVSWITCH Gen5"
