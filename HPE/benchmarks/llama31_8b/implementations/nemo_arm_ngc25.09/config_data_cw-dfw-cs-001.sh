export VERIFY_MOUNTS=0

# data path on global file system
export RO_DATA="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/data"

# get checkpoint from lustre since it is touched only during init
export LOAD_CHECKPOINTS_PATH="${RO_DATA}/llama31/nemo-ckpt/"
export LOAD_CHECKPOINT="/load_checkpoints/405b"

export FAST_LOCAL_NVME="/raid/scratch/mlperft-llama31-v50"
export DATADIR="${FAST_LOCAL_NVME}/c4"
export SLOW_DATADIR="${RO_DATA}/c4"

export USER_DROPCACHE="${RO_DATA}/mlperf-common/client/dropcache"

# turn off hang monitor because I worry it may be causing more problems than it is solving
export HANG_MONITOR_TIMEOUT=0

# need these global tmp directories during init for syncrhonizing these
# indices, so this needs to be on global file system, and writable
export RW_TMP="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/tmp"
export GLOBAL_TMP_NPY_INDEX_DIR="${RW_TMP}/tmp_npy_index"
export GLOBAL_TMP_CHECKPOINTS_DIR="${RW_TMP}/tmp_checkpoints"

# this system has an extended ephemeral port range (9000-65000), which
# conflicts with our default choice of 29500, so need to choose a MASTER_PORT
# outside that extended range
export MASTER_PORT="65500"

export SLURM_MPI_TYPE=pmi2
export NCCL_IB_SL=1

# each of these nccl tests takes a couple minutes at scale, and the init time
# at scale has become "excrutiating" so turning these off
export NCCL_LLM_TEST=0
export NCCL_TEST=0

# disabled because the cs-cw-dfw team is not stress testing IPoIB
# # workaround for dns errors from pytorch:
# export USE_IPOIB=1

# commented out as it is not needed after cs-cw-dfw FW update on 2/28/24
# settings given to me by Vipin Sirohi:
# export NCCL_IB_HCA=^mlx5_3,mlx5_4
# export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

# commented out as it is not needed after cs-cw-dfw FW update on 2/28/24
# we believe this setting is required to work around
# https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=4302831&cmtNo=
# export NCCL_SOCKET_IFNAME=ibp101s0

# now we need to set it this way to work around the (now incorrect) workaround applied to Enroot
export NCCL_SOCKET_IFNAME=enp90s0f0np0

# make sure MPI is _not_ using UCC
export OMPI_MCA_coll=^ucc

# Turn off https://docs.nvidia.com/networking/display/HPCXv29/Unified+Communication+-+X+Framework+Library#UnifiedCommunicationXFrameworkLibrary-include:FUSE-basedToolInstrumentationandMonitoringFUSE-basedTool
export UCX_VFS_ENABLE=no

export GPU_ARCH=h100

# We were setting these in May due to https://nvbugs/4036883
#export NCCL_IGNORE_CPU_AFFINITY=1
#export NCCL_PROTO=^LL128
#export NCCL_IB_PCI_RELAXED_ORDERING=0
#export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=0

# turn off coredumps due to a mkdir permissions error in older containers
export ATTEMPT_CUDA_GDB_CORE_DUMP=0

# commented out as it is not needed after cs-cw-dfw FW update on 2/28/24
# Fix to deal with GLOO init error due to transient ethernet issues
# export GLOO_SOCKET_IFNAME=ibp101s0

# JET
export JET_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh"

############# Cosmetic stuff for submission system.json files ###############
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="Available cloud"
export MLPERF_SYSTEM_NAME="Eos-dfw (@MLPERF_SYS_SIZE@x NVIDIA HGX H100)"
export MLPERF_FRAMEWORK="NVIDIA NeMo Framework Release 25.01"
export MLPERF_FRAMEWORK_SHORT_NAME="ngc25.01_nemo"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="2x 1.92TB NVMe SSD + 8x 3.84TB NVMe U.2"
export MLPERF_HOST_NETWORKING="Storage+Management: 1x NVIDIA BlueField 2 DPU 100Gb/s, Compute: 8x NVIDIA ConnectX-7 IB NDR 400Gb/s"
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA H100-SXM5-80GB"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="HBM3"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLINK Gen4 900 GB/s + NVSWITCH Gen3"
