export SLURM_MPI_TYPE=pmi2
export UCX_TLS=self,tcp

# the following socket ifnames are dlcluster & GB200 specific:
export GLOO_SOCKET_IFNAME=enP5p9s0
export TP_SOCKET_IFNAME=enP5p9s0

# 8b checkpoint test
export LOAD_CHECKPOINT=""
export CHECKPOINT_NAME="${CHECKPOINT_NAME:-16n_torch_dist}"

export TOKENIZER="/raid/dldata/mlperft/c4/mixtral-tokenizer"
export PREPROC_DATA="/raid/dldata/mlperft/c4"

# Mock dataset
export CLEAR_CACHES=0
export GET_MOUNT_INFO=0 # does not do container mount check for now

# need these global tmp directories during init for synchronizing these
# indices, so this needs to be on global file system, and writable
#export GLOBAL_TMP_NPY_INDEX_DIR="/lustre/fsw/coreai_mlperf_training/gpt3/tmp_npy_index"
#export GLOBAL_TMP_CHECKPOINTS_DIR="/lustre/fsw/coreai_mlperf_training/gpt3/tmp_checkpoints"

export NEMO_RESULTS_IN_TMP=1
export GPU_ARCH=b200
export NCCL_IB_SL=1

# JET
export JET_DIR="/mnt/nvdl/usr/michalm/mlperf/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh "

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized/-/blob/main"

############# Cosmetic stuff for submission system.json files ###############
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="Preview"
export MLPERF_SYSTEM_NAME="gb200_preview"
export MLPERF_SYSJSON_SYSNAME_INCLUDE_NUM_NODES=1
export MLPERF_FRAMEWORK="NVIDIA NEMO"
export MLPERF_FRAMEWORK_SHORT_NAME="nemo_preview"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY=""
export MLPERF_HOST_NETWORKING=""
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA Blackwell GPU (GB200)"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION=""
export MLPERF_ACCELERATOR_INTERCONNECT=""
