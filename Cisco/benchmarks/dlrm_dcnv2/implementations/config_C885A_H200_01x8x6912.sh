## DL params
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1 
export PMIX_GDS_MODULE=^ds12
export PMIX_MCA_gds=^ds12
export NCCL_NET_PLUGIN=none
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_CHECHSUM_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

## DL params
export RUN_SCRIPT="train.py"
export BATCHSIZE=65536
export BATCHSIZE_EVAL=262144
export LEARNING_RATE=0.004
export USE_MIXED_PRECISION=true
export SCALER=16348
export SHARDING_PLAN=auto
export MEM_COMM_BW_RATIO=11
export GEN_LOSS_SUMMARY=true
export MINIMUM_TRAINING_TIME=10
export DP_SHARDING_THRESHOLD=0.008

## System run parms
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=10

## Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1320
  WALLTIME_RUNANDTIME=$(expr ${WALLTIME_RUNANDTIME} + ${WALLTIME_RUNANDTIME} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1665
  WALLTIME_RUNANDTIME=$(expr ${WALLTIME_RUNANDTIME} + ${WALLTIME_RUNANDTIME} / 3) # 33% longer walltime
fi
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

#Performance paramenter
#export DLRM_BIND="numactl --membind=0,1"
export DLRM_BIND="./bindpcie --cpu=bind_cpu_topology.sh --mem=bind_mem_topology.sh --ib=single"
#Mandatory param NCCL_IB_GID_INDEX=3
export NCCL_IB_GID_INDEX=3