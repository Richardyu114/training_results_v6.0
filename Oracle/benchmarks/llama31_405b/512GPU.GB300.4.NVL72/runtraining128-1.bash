export WALLTIME=600
export CONT=/lfs/mlperf/sqsh/nvcr.io+nvdlfwea+mlperftv60+llama31_405b-arm+20260511.sqsh
#export DATADIR=/lfs/mlperf/data/dataset-LLAMA-3.1-405B
export DATADIR=/lfs/mlperf/scratch/llama3.1-405b/data
export LOAD_CHECKPOINTS_PATH=/lfs/mlperf/scratch/llama3.1-405b
export LOAD_CHECKPOINT="/load_checkpoints/405b"
export LOGDIR=/lfs/mlperf/log/LLAMA-3.1-405B/log/128
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/
module load mpi/openmpi/5.0.8-gcc
source config_GB300_128x4x56xtp2pp8cp2_cg_fp4.sh
source /lfs/mlperf/common/nccl.env
#export NODELIST=gpu-2a458cee-[0-5,7-11,13-14,16-17],gpu-17b928d4-[0-4,6-9,11-12,14-17],gpu-79d641e1-[0-3,6-9,11-12,14-16],gpu-ae65613a-[0,2-3,5-17],gpu-bcafd65f-[0-1,3-12,14-17],gpu-d6819806-[0-11,13-15,17],gpu-e0a94389-[1,3-6,9,12-13,15-17],gpu-f9730b5a-[0-4,6-14,17],gpu-f44283e4-[2-9,11-12,14]
#export NODELIST=gpu-f44283e4-[5,7-8,3,6,14,9,4,2,12],gpu-17b928d4-[1-2,4,12,6,9,7-8,14-15,17,0,16,3,11],gpu-d6819806-[14,0-1,17,2,13,10,15,11,8,6,4-5,9,7,3],gpu-4fab1453-[3,11,8,14,6-7,2,0,17,15,12,16,4,10,5,13],gpu-2a458cee-[0,8,14,2,10,13,7,4-5,9,1,16-17,11,3],gpu-bcafd65f-[4,10,5,12,16,8,7,11,15,0,6,9,1,3,17,14],gpu-f9730b5a-[10,9,2,6,3,8,11,1,12],gpu-79d641e1-[9,2,8,16,3,12,15,7,6,1,14,0,11,17,10],gpu-ae65613a-[12,11,15,6,2,14,5,3,0,9,8,17,7,10,13,16]

if [ -z "$NODELIST" ]; then
    echo "No nodelist provided → using scheduler default"
    NODELISTENV=""
else
    echo "Using provided nodes: $NODELIST"
    NODELISTENV="--nodelist $NODELIST"
fi
sbatch ${NODELISTENV}  -N $DGXNNODES -t $WALLTIME run.sub
