#!/bin/bash

cd ../hugectr
export CONT=./nvdlfwea+mlperftv60+dlrm_dcnv2-amd+.sqsh
source config_G894-SD3_1x8x8192.sh
export DATADIR=/path/to/datadir
export DATADIR_VAL=/path/to/datadir_val
export MODEL=/path/to/model
export LOGDIR=/path/to/logdir
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G894-SD3-AAX7

sbatch -N $DGXNNODES -t $WALLTIME run.sub
