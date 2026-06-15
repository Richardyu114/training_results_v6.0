#!/bin/bash

# This script should be called on a compute node *inside* a container.
# Triggers a core dump in the application.
# Assumes that the application was run with CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1.
#
# Expecting the following env variables to be set externally:
#   - CUDA_COREDUMP_PIPE - name of pipes created for user-triggered core dump

set -x

PIPE_DIR=$(dirname $CUDA_COREDUMP_PIPE)
echo "$0: dumping GPU core to ${CUDA_COREDUMP_FILE} by triggering pipes from ${PIPE_DIR}"

for corepipe in ${PIPE_DIR}/*; do
  echo 1 >> $corepipe &
done

echo "Dumping cores done on host $(hostname)"
