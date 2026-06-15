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

export NEXP=10

: "${DATADIR:?DATADIR not set}"
: "${MODEL:?MODEL not set}"
: "${CONT:?CONT not set}"

export WARMUP=True
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export TP_COMM_OVERLAP=True
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True
export MC_TP_OVERLAP_RS_DGRAD=True
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export NVTE_RS_STRIDED_ATOMIC=2
export LORA_A2A=1

export POSSIBLE_USER_WARNINGS=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=32
export HYDRA_FULL_ERROR=1

export PP=1
export SP=1

# other
export MBS=1
export VAL_CHECK_INTERVAL=384

export PMIX_MCA_gds=hash
