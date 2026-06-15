source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_16x8x1xtp2pp1cp1ep4.sh

export DGXNNODES=8

export LR=0.0004
export LR_WARMUP_STEPS=512
export VAL_CHECK_INTERVAL=384

export WALLTIME_RUNANDTIME=120
export WALLTIME=$((5 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 5)))

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//')
