source $(dirname ${BASH_SOURCE[0]})/config_GB200_2x4x6912.sh

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
