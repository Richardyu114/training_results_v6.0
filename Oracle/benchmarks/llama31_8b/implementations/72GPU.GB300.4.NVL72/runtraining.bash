export WALLTIME=600
export CONT=/lfs/mlperf/sqsh/nvcr.io+nvdlfwea+mlperftv60+llama31_8b-arm+20260507.sqsh
export DATADIR=/lfs/mlperf/data/dataset-LLAMA-3.1-8B-GPT-OSS-20B
export LOGDIR=/lfs/mlperf/log/LLAMA-3.1-8B/2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/
module load mpi/openmpi/5.0.8-gcc
source config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh
source /lfs/mlperf/common/nccl.env

if [ -z "$NODELIST" ]; then
    echo "No nodelist provided → using scheduler default"
    NODELISTENV=""
else
    echo "Using provided nodes: $NODELIST"
    NODELISTENV="--nodelist $NODELIST"
fi
sbatch ${NODELISTENV}  -N $DGXNNODES -t $WALLTIME run.sub
