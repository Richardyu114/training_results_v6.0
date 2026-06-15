export VERIFY_MOUNTS=0

# data and checkpoint
RO_DATA="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/data"
export FAST_LOCAL_NVME="/raid/scratch/coreai_mlperf_training"

export LOAD_CHECKPOINTS_PATH="${RO_DATA}/llama31_405b/nemo_ckpt"
export SLOW_DATADIR="${RO_DATA}/c4"

# LOAD_CHECKPOINT is the mount path inside the container, thus shouldn't be
# specified here (doesn't actually need to be configurable at all, but if
# configured, shouldn't be in the cluster config)
export LOAD_CHECKPOINT="/load_checkpoints/405b"

export DATADIR="${FAST_LOCAL_NVME}/c4"

export USER_DROPCACHE="${RO_DATA}/bin/dropcache"

export HANG_MONITOR_TIMEOUT=0

# need these global tmp directories during init for synchronizing these
# indices, so this needs to be on global file system, and writable
#export GLOBAL_TMP_NPY_INDEX_DIR="/lustre/fsw/coreai_mlperf_training/gpt3/tmp_npy_index"
#export GLOBAL_TMP_CHECKPOINTS_DIR="/lustre/fsw/coreai_mlperf_training/gpt3/tmp_checkpoints"

export NCCL_TEST=1

# this system has an extended ephemeral port range (9000-65000), which
# conflicts with our default choice of 29500, so need to choose a MASTER_PORT
# outside that extended range
export MASTER_PORT="65500"

export NEMO_RESULTS_IN_TMP=1
export GPU_ARCH=gb200
export NCCL_IB_SL=1
export SLURM_MPI_TYPE=pmix
export UCX_TLS=self,tcp

# DLSIM
export DLSIM_REPORT_DIR="/lustre/share/coreai_mlperf_training/dlsim"
export DLSIM_DEVICE="Tesla_GB200_SXM_189GB"
export DLSIM_METHODOLOGY="blackwell.optimistic"

# JET
export JET_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh "

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized/-/blob/main"

############# Cosmetic stuff for submission system.json files ###############
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="Available on-premise"
export MLPERF_SYSTEM_NAME='oci-hsg (@MLPERF_SYS_SIZE@x NVIDIA GB200 NVL72)'
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA Blackwell GPU (GB200)"
export MLPERF_FRAMEWORK="NVIDIA NeMo Framework Release 25.09"
export MLPERF_FRAMEWORK_SHORT_NAME="ngc25.09_nemo"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="8x 3.5TB NVMe E1.s"
export MLPERF_HOST_NETWORKING="4x ConnectX-7 IB NDR 400 Gb/s"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="HBM3e"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLINK Gen5 1800 GB/s + NVSWITCH Gen5"
