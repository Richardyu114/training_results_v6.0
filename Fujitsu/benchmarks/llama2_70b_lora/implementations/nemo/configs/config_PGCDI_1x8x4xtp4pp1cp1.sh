#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# overwrite common parameters
export TP_COMM_OVERLAP=True
export UB_SKIPMC=1

# remove specific environment variables
#unset NCCL_MIN_P2P_NCHANNELS
#unset NCCL_MIN_CTAS
#unset NCCL_NCHANNELS_PER_NET_PEER
#unset MC_TP_OVERLAP_AG
#unset MC_TP_OVERLAP_RS
#unset MC_TP_OVERLAP_RS_DGRAD

# hyperparameters
export MAX_STEPS=800
export LR=0.0005
export MINIBS=4
export TP=4
export CP=1
#export TP_COMM_OVERLAP=1
export VBOOST_VALUE=1
export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1
export SKIP_EVALS=3

# system parameters
export DGXNNODES=1
export WALLTIME_RUNANDTIME=300
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
