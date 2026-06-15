source $(dirname ${BASH_SOURCE[0]})/config_common.sh
## DL params
export RUN_SCRIPT="train.py"
export BATCHSIZE=135168
export BATCHSIZE_EVAL=1048576
export LEARNING_RATE=0.0034
export USE_MIXED_PRECISION=true
export SCALER=20480
export SHARDING_PLAN=hier_auto
export MEM_COMM_BW_RATIO=160
export GEN_LOSS_SUMMARY=true
export MINIMUM_TRAINING_TIME=10
export DP_SHARDING_THRESHOLD=0.0125

## System run parms
export DGXNNODES=2
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=15
#export SBATCH_NETWORK=sharp
#export NCCL_COLLNET_ENABLE=1
#export SHARP_ALLOW_SM_PORT=1
## Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1320
  WALLTIME_RUNANDTIME=$(expr ${WALLTIME_RUNANDTIME} + ${WALLTIME_RUNANDTIME} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1665
  WALLTIME_RUNANDTIME=$(expr ${WALLTIME_RUNANDTIME} + ${WALLTIME_RUNANDTIME} / 3) # 33% longer walltime
fi
# Override NVLS from config_common.sh - causes AUC NCCL warm-up hang on multi-node XE9780
export NCCL_NVLS_ENABLE=1
# Additional NCCL settings to fix AUC warm-up hang on multi-node
#export NCCL_P2P_DISABLE=PIX
export NCCL_IB_HCA=^smi1
export NCCL_IB_TIMEOUT=50
export NCCL_MAX_NRINGS=16
export NCCL_ALGO=Ring,Tree
export WALLTIME=UNLIMITED
