source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_8x8x36xtp8pp8cp2.sh

export DGXNNODES=16
export VAL_CHECK_INTERVAL=192
export LR=0.0007
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
