## data path
export SLOW_DATADIR="/lustre/fsw/coreai_mlperf_training/data/lora/gov_report"
export DATADIR="/raid/scratch/lora/gov_report"
# export MODEL="/lustre/fsw/coreai_mlperf_training/data/lora/model"
export MODEL=/lustre/share/coreai_mlperf_training/user/mfutrega/ckpt_torch_dist
export SLURM_MPI_TYPE=pmix
export UCX_TLS=self,tcp
export GPU_ARCH="b"

# DLSIM
export DLSIM_DEVICE="Tesla_GB200_SXM_189GB"
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
export MLPERF_SYSTEM_NAME='Tyche (@MLPERF_SYS_SIZE@x NVIDIA GB200 NVL72)'
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA Blackwell GPU (GB200)"
export MLPERF_FRAMEWORK="NVIDIA NeMo Framework Release 25.09"
export MLPERF_FRAMEWORK_SHORT_NAME="ngc25.09_nemo"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="8x 3.5TB NVMe E1.s"
export MLPERF_HOST_NETWORKING="4x ConnectX-7 IB NDR 400 Gb/s"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="HBM3e"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLINK Gen5 1800 GB/s + NVSWITCH Gen5"
