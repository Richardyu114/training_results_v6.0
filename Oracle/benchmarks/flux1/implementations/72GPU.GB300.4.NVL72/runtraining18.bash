export WALLTIME=600
export CONT=/lfs/mlperf/sqsh/sd+nvdlfwea+mlperftv60+flux1-arm+20260511.sqsh
export DATAROOT=/lfs/mlperf/data/dataset-FLUX1/energon
export LOGDIR=/lfs/mlperf/log/FLUX1/18
module load mpi/openmpi/5.0.8-gcc
source /lfs/mlperf/common/optnccl.env
source config_GB300_18x04x32.sh
if [ -z "$NODELIST" ]; then
    echo "No nodelist provided → using scheduler default"
    NODELISTENV=""
else
    echo "Using provided nodes: $NODELIST"
    NODELISTENV="--nodelist $NODELIST"
fi
sbatch ${NODELISTENV}  -N $DGXNNODES -t $WALLTIME run.sub
