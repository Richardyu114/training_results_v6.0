#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

set -eux

nsys_report_types=("nvtx_kern_sum" "nvtx_gpu_proj_sum")
nsys_report_filetype=csv

cd /workspace/llm_auto_breakdown/correlate

# Extract profiled ranks from environment variable
profile_ranks=($(echo "$PROFILE_RANKS" | tr "," "\n"))

# Perform breakdown only if there is a profile for my rank
if [[ " ${profile_ranks[@]} " =~ " ${SLURM_PROCID} " ]]; then
    nsys_reports=()
    nsys_prefix_rank="${NSYS_PREFIX}${SLURM_PROCID}"
    nsys_profile="${nsys_prefix_rank}${NSYS_SUFFIX}"

    for nsys_report_type in "${nsys_report_types[@]}"; do
        nsys stats --report $nsys_report_type --timeunit usec -f $nsys_report_filetype -o . $nsys_profile
        nsys_reports+=("$nsys_prefix_rank"_"$nsys_report_type"."$nsys_report_filetype")
    done

    dlsim_args=()
    if [[ -n "${DLSIM_REPORT_CONT:-}" ]]; then
        dlsim_args=(--dlsim "$DLSIM_REPORT_CONT")
    fi
    python3 correlate.py --benchmark llm-llama31 --mode read_summary --silicon "${nsys_reports[@]}" "${dlsim_args[@]}" --output "$nsys_prefix_rank"_correlate.csv
fi
