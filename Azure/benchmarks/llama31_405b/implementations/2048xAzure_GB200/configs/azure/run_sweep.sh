#!/bin/bash
# ==============================================================================
# Run a single experiment with variable DP and MINIBS.
# Overrides MINIBS, DGXNNODES (=DP*16), and LOGDIR on the fly.
#
# Usage:
#   source configs/run_sweep.sh <DP> <MINIBS>
#   source configs/run_sweep.sh 8 144
# ==============================================================================

if [[ $# -ne 2 ]]; then
    echo "Usage: source $0 <DP> <MINIBS>"
    echo "  DP      = data parallel replicas (DGXNNODES = DP * 16)"
    echo "  MINIBS  = mini-batch size per DP rank"
    return 1 2>/dev/null || exit 1
fi

_DP=$1
_MINIBS=$2
_DGXNNODES=$(( _DP * 16 ))
_GBS=$(( _MINIBS * _DP ))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${SCRIPT_DIR}/config_GB200_640x4x24xtp4pp8cp2_cg_fp4_mrc.sh"

# Source the MRC config (sets MRC env + all common vars)
source "${BASELINE}"

# Override with sweep values
export MINIBS=${_MINIBS}
export DGXNNODES=${_DGXNNODES}
export LOGROOT=$(realpath "$(dirname ${BASH_SOURCE[0]})/../../logs_DP_${_DP}_MINIBS_${MINIBS}/")
export LOGDIR="${LOGROOT}/$(date +%y%m%d%H%M%S%N)"

mkdir -p "${LOGDIR}"

echo "================================================================================"
echo "Experiment: DGXNNODES=${DGXNNODES} MINIBS=${MINIBS} DP=${_DP} GBS=${_GBS}"
echo "================================================================================"
echo "  LOGDIR=${LOGDIR}"
