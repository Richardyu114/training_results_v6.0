#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_GB200_2x4x1xtp1pp1cp1.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_apilogs.sh

export DGXNNODES=1
export VAL_CHECK_INTERVAL=80  # MAX_STEPS (20) * GBS (2)
export WALLTIME_RUNANDTIME=50
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
export BATCHSIZE=${MBS}
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# Set clocks
if [[ "${SET_GPU_CLK:-1}" == "1" ]]; then
    export MAX_GPU_CLK=1845
fi
