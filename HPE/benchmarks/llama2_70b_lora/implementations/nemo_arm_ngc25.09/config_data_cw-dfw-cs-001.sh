## data path
export DATADIR="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/data/lora/gov_report"
export MODEL="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/data/lora/model"

### VBOOST not enabled on this system
export VBOOST=0

# this system has an extended ephemeral port range (9000-65000), which
# conflicts with our default choice of 29500, so need to choose a MASTER_PORT
# outside that extended range
export MASTER_PORT="65500"


export SLURM_MPI_TYPE=pmi2
export NCCL_IB_SL=1
export UCX_TLS=self,tcp

# now we need to set it this way to work around the (now incorrect) workaround applied to Enroot
export NCCL_SOCKET_IFNAME=enp90s0f0np0

# make sure MPI is _not_ using UCC
export OMPI_MCA_coll=^ucc

# Turn off https://docs.nvidia.com/networking/display/HPCXv29/Unified+Communication+-+X+Framework+Library#UnifiedCommunicationXFrameworkLibrary-include:FUSE-basedToolInstrumentationandMonitoringFUSE-basedTool
export UCX_VFS_ENABLE=no

# JET
export JET_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/jet"
export JET_UPLOAD="jet logs upload output.zip"
export JET_CREATE="jet logs create output.zip --fill-gpu --fill-cpu --fill-system --fill-libraries --data user=${USER} --data workload.maintainers[]=${USER} --data type=workload --data workload.type=custom --data workload.spec.script=run_and_time.sh"

############# Cosmetic stuff for submission system.json files ###############
export MLPERF_SUBMITTER="NVIDIA"
export MLPERF_STATUS="Available cloud"
export MLPERF_SYSTEM_NAME="Eos-dfw"
export MLPERF_FRAMEWORK="NVIDIA NeMo Framework Release 25.04"
export MLPERF_FRAMEWORK_SHORT_NAME="ngc25.04_nemo"

export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="2x 1.92TB NVMe SSD + 8x 3.84TB NVMe U.2"
export MLPERF_HOST_NETWORKING="Storage+Management: 1x NVIDIA BlueField 2 DPU 100Gb/s, Compute: 8x NVIDIA ConnectX-7 IB NDR 400Gb/s"
export MLPERF_ACCELERATOR_MODEL_NAME="NVIDIA H100-SXM5-80GB"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="HBM3"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLINK Gen4 900 GB/s + NVSWITCH Gen3"
