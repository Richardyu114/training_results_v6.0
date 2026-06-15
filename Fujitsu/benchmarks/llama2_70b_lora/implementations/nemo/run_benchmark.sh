#! /bin/bash
#

# setup slurm command path
SLURM_HOME=/opt/slurm-25-11-5-1
export PATH=${SLURM_HOME}/bin:${PATH}

OMPI_HOME=/opt/ompi-5.0.10
#export LD_LIBRARY_PATH=${OMPI_HOME}/lib:${LD_LIBRARY_PATH}

PMIX3_HOME=/opt/pmix-3.2.5/lib
export LD_LIBRARY_PATH=${PMIX3_HOME}/lib:${LD_LIBRARY_PATH}

# setup logdir
logname=log_$(date +%Y%m%d_%H%M%S)
logdir=./logs/${logname}

mkdir ${logdir}

export DATADIR="/mnt/data4/work/llama2-v60/gov_report" # set your </path/to/dataset>
export MODEL="/mnt/data4/work/llama2-v60/model"  # set your </path/to/dataset>
export LOGDIR="$(realpath ${logdir})"  # set the place where the output logs will be saved
export CONT=/home/notsu/training_v6.0/sqsh_images/llama2_70b_lora_mlc6.0.sqsh
export NEXP=10
source configs/config_PGCDI_1x8x4xtp4pp1cp1.sh

sbatch -N $DGXNNODES -t $WALLTIME -o outputs/slurm-%j.out run.sub  # you may be required to set --account and --partition here
