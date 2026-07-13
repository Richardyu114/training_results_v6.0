#!/bin/bash
# Two-node (2x8 GPU) Llama 3.1 8B baseline for MI325X.

_TWO_NODE_PLATFORM=MI325X
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/config_common_2x8x1.sh" || {
    _config_rc=$?
    return "${_config_rc}" 2>/dev/null || exit "${_config_rc}"
}
unset _TWO_NODE_PLATFORM
