export WALLTIME=600
export WALLTIME_RUNANDTIME=30
export CONT=/lfs/mlperf/sqsh/nvcr.io+nvdlfwea+mlperftv60+llama2_70b_lora-arm+20260507.sqsh
export DATADIR=/lfs/mlperf/data/dataset-LLAMA-2-70B-LoRA/gov_report
export MODEL=/lfs/mlperf/data/dataset-LLAMA-2-70B-LoRA/model
export LOGDIR=/lfs/mlperf/log/LLAMA-2-70B-LoRA/log/18
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/
export NCCL_TEST=0
module load mpi/openmpi/5.0.8-gcc
source config_GB300_18x4x1xtp1pp1cp8.sh
source /lfs/mlperf/common/optnccl.env
export NODELIST=gpu-e0a94389-[7,10],gpu-4fab1453-[3,11,8,14,6-7,2,0,17,15,12,16,4,10,5,13]

if [ -z "$NODELIST" ]; then
    echo "No nodelist provided → using scheduler default"
    NODELISTENV=""
else
    echo "Using provided nodes: $NODELIST"
    NODELISTENV="--nodelist $NODELIST"
fi
sbatch ${NODELISTENV}  -N $DGXNNODES -t $WALLTIME run.sub
