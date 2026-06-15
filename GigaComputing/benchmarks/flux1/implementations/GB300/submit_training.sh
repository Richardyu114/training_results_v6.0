#!/bin/bash
cd /sharedata/flux1
export CONT=/sharedata/flux1/nvdlfwea+mlperftv60+flux1-arm+20260423.sqsh
source config_GB300_18x04x32.sh
export DATAROOT="/data/mlperf_training/dataset/flux1/energon"
export LOGDIR="/sharedata/flux1/results"
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=enP22p3s0f1np1
export UCX_NET_DEVICES=enP22p3s0f1np1
export UCX_TLS=tcp
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=DLB2-CB3
sbatch -N $DGXNNODES --time=UNLIMITED --gres=gpu:4 run.sub
