source $(dirname ${BASH_SOURCE[0]})/config_GB200_2x4x4xtp1pp1cp1_8b.sh

export MAX_STEPS=720
export LIMIT_VAL_BATCHES=40
export VAL_CHECK_INTERVAL=360
export FORCE_SUCCESS_STATUS=1

export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
