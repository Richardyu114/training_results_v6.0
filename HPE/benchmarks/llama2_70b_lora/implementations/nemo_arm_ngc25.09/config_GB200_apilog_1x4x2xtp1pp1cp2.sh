#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_GB200_1x4x4xtp1pp1cp2.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_apilogs.sh

export VAL_CHECK_INTERVAL=160  # MAX_STEPS (20) * GBS (8)
export WALLTIME_RUNANDTIME=50
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
export BATCHSIZE=${MBS}
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# Set clocks
if [[ "${SET_GPU_CLK:-1}" == "1" ]]; then
    export MAX_GPU_CLK=1845
fi
