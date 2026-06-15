export WALLTIME=600
export CONT=/lfs/mlperf/sqsh/sd+nvdlfwea+mlperftv60+flux1-arm+20260511.sqsh
export DATAROOT=/lfs/mlperf/data/dataset-FLUX1/energon
export LOGDIR=/lfs/mlperf/log/FLUX1/8
#source config_GB300_18x04x32.sh  # select config and source it
module load mpi/openmpi/5.0.8-gcc
source config_GB300_08x04x32.sh
#
#source config_GB300_18x04x32.sh
#NODELIST=gpu-4fab1453-[0,2-8,10-17],gpu-e0a94389-[4,6]
export NODELIST=gpu-4fab1453-0,gpu-4fab1453-12,gpu-4fab1453-3,gpu-4fab1453-17,gpu-4fab1453-16,gpu-4fab1453-11,gpu-17b928d4-11,gpu-4fab1453-8,gpu-4fab1453-10,gpu-4fab1453-6,gpu-17b928d4-6,gpu-17b928d4-12,gpu-4fab1453-7,gpu-4fab1453-14,gpu-17b928d4-7,gpu-4fab1453-15,gpu-17b928d4-17,gpu-17b928d4-0,gpu-4fab1453-13,gpu-17b928d4-3,gpu-17b928d4-16,gpu-4fab1453-2,gpu-17b928d4-8,gpu-17b928d4-2,gpu-17b928d4-14,gpu-17b928d4-9,gpu-4fab1453-5,gpu-17b928d4-1,gpu-79d641e1-0,gpu-4fab1453-4,gpu-17b928d4-4,gpu-17b928d4-15,gpu-79d641e1-1,gpu-79d641e1-7,gpu-d6819806-4,gpu-79d641e1-6,gpu-d6819806-5,gpu-79d641e1-3,gpu-79d641e1-12,gpu-d6819806-9,gpu-d6819806-13,gpu-79d641e1-2,gpu-79d641e1-8,gpu-d6819806-1,gpu-79d641e1-17,gpu-d6819806-3,gpu-79d641e1-9,gpu-79d641e1-16,gpu-d6819806-17,gpu-79d641e1-15,gpu-79d641e1-11,gpu-79d641e1-10,gpu-d6819806-6,gpu-d6819806-7,gpu-d6819806-0,gpu-d6819806-2,gpu-79d641e1-14,gpu-d6819806-11,gpu-e0a94389-1,gpu-d6819806-14,gpu-e0a94389-3,gpu-d6819806-8,gpu-d6819806-10,gpu-d6819806-15,gpu-e0a94389-4,gpu-e0a94389-6,gpu-e0a94389-12,gpu-e0a94389-9,gpu-e0a94389-5,gpu-e0a94389-13,gpu-f44283e4-2,gpu-e0a94389-16,gpu-e0a94389-10,gpu-e0a94389-15,gpu-e0a94389-17,gpu-e0a94389-14,gpu-f44283e4-4,gpu-f44283e4-9,gpu-e0a94389-7,gpu-f44283e4-12,gpu-f44283e4-3,gpu-f44283e4-6,gpu-f44283e4-14,gpu-f44283e4-8,gpu-f44283e4-7,gpu-f44283e4-1,gpu-f44283e4-11
if [ -z "$NODELIST" ]; then
    echo "No nodelist provided → using scheduler default"
    NODELISTENV=""
else
    echo "Using provided nodes: $NODELIST"
    NODELISTENV="--nodelist $NODELIST"
fi
sbatch ${NODELISTENV}  -N $DGXNNODES -t $WALLTIME run.sub
