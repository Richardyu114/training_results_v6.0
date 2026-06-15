export DATADIR=/raid/dldata/mlperft/text/govreport/mlperf
# TODO: Upadate path to new checkpoint format
export MODEL=/raid/dldata/mlperft/model/llama2_70b_pyt/nemo_mode-pretrain/mlperf_bf16

export ENABLE_IB_BINDING=0

# dlcluster doesn't seem to set MELLANOX_VISIBLE_DEVICES by default
#export MELLANOX_VISIBLE_DEVICES=0,1,2,3,6,7,8,9
export SLURM_MPI_TYPE=pmi2
#export NCCL_NET_HCA=ibp24s0,ibp64s0,ibp79s0,ibp94s0,ibp154s0,ibp192s0,ibp206s0,ibp220s0
#export NCCL_IBEXT_DISABLE=1
export UCX_TLS=self,tcp
#export NCCL_IB_DISABLE=1
export CLEAR_CACHES=0

# the following socket ifnames are dlcluster & GB200 specific:
export GLOO_SOCKET_IFNAME=enP5p9s0
export TP_SOCKET_IFNAME=enP5p9s0

# JET
export JET_DIR="/mnt/nvdl/usr/michalm/mlperf/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh"

# Readme prefix for JSON file
export README_PREFIX="https://gitlab-master.nvidia.com/dl/mlperf/optimized/-/blob/main"

############# Cosmetic stuff for submission system.json files ###############
# NOTE: not updated after moving from GH200 to GB200
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="onprem"
export MLPERF_SYSTEM_NAME="NVIDIA H200 (8x H200-SXM-141GB-CTS)"
export MLPERF_SYSJSON_SYSNAME_INCLUDE_NUM_NODES=0
export MLPERF_FRAMEWORK="NVIDIA NeMo Framework Release 24.09"
export MLPERF_FRAMEWORK_SHORT_NAME="ngc24.09_nemo"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="2x 1.75TB NVMe SSD + 8x3.5TB NVMe"
export MLPERF_HOST_NETWORKING="Storage: 2x ConnectX-7 IB NDR 400Gb/s, Compute: 8x ConnectX-7 IB NDR 400Gb/s, Management: 100Gb/s Ethernet NIC"
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA H200-SXM5-141GB-CTS"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="HBM3e"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLINK Gen4 900 GB/s + NVSWITCH Gen3"
