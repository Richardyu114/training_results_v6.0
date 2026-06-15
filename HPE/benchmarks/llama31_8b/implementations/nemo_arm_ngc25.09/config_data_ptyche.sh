# 8b checkpoint test
export LOAD_CHECKPOINTS_PATH="/lustre/fsw/coreai_mlperf_training/data/llm/llama31-405b-version/nemo-ckpt"
export LOAD_CHECKPOINT="/load_checkpoints/405b"

export DATADIR="/lustre/share/coreai_mlperf_training/data/c4"
export GET_MOUNT_INFO=0 # does not do container mount check for now

# need these global tmp directories during init for syncrhonizing these
# indices, so this needs to be on global file system, and writable
#export GLOBAL_TMP_NPY_INDEX_DIR="/lustre/fsw/coreai_mlperf_training/gpt3/tmp_npy_index"
#export GLOBAL_TMP_CHECKPOINTS_DIR="/lustre/fsw/coreai_mlperf_training/gpt3/tmp_checkpoints"

export NEMO_RESULTS_IN_TMP=1
export GPU_ARCH=gb200
export NCCL_IB_SL=1
export SLURM_MPI_TYPE=pmix
export UCX_TLS=self,tcp


# DLSIM
export DLSIM_DEVICE="Tesla_GB200_SXM_189GB"
export DLSIM_METHODOLOGY="blackwell.optimistic"

# NSYS: Specify the name of the metrics file to use when capturing metrics
export NSYS_METRICS_SET="gb10x"

# JET
export JET_DIR="/project/coreai_mlperf_training/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh "

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized/-/blob/main"

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
