# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export HYDRA_FULL_ERROR=1
export TQDM_DISABLE=True
export DISABLE_PRINT=${DISABLE_PRINT:-True} # filter out nemo prints. Set to False to debug.

export MODEL=${MODEL:-schnell}
export DATA=${DATA:-cc12m}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export EXP_NAME=${EXP_NAME:-flux-train-$(date +%y%m%d%H%M%S%N)}
# Validation interval in samples
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}

export GRAD_REDUCE_IN_FP32=${GRAD_REDUCE_IN_FP32:-False}

export WARMUP_ENABLED=${WARMUP_ENABLED:-True}
export WARMUP_TRAIN_STEPS=${WARMUP_TRAIN_STEPS:-2}
export WARMUP_VALIDATION_STEPS=${WARMUP_VALIDATION_STEPS:-2}

export HF_HUB_OFFLINE=1 # disable network requests for HF transfers

export NCCL_NVLS_ENABLE=1
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0
export NCCL_PAT_ENABLE=1  # Pattern-Aware Transport

# DDP optimization
export BUCKET_SIZE=${BUCKET_SIZE:-512000000}  # 512MB buckets
export AVERAGE_IN_COLLECTIVE=${AVERAGE_IN_COLLECTIVE:-True}
export GRADIENT_ACCUMULATION_FUSION=${GRADIENT_ACCUMULATION_FUSION:-True}  # Fuse .add_() into backward GEMM

export MAX_AUTOTUNE_JIT_FUSER=${MAX_AUTOTUNE_JIT_FUSER:-1}  # max-autotune for @jit_fuser AdaLN ops
export DISABLE_CDT=${DISABLE_CDT:-0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_cache}  # Node-local: avoids Lustre stale file handles

# Data loading
export NUM_WORKERS=${NUM_WORKERS:-4}  # Fewer workers reduces CPU thread contention

export MODEL_TFLOP_PER_SAMPLE=20.3
