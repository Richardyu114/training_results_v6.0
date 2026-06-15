#!/bin/bash

source config_MI350X_1x8x1.sh

export DATADIR=
export LOGDIR=
export CONT=
export LOGFILE="llama70b-lora_$(date +%Y-%m-%d_%H-%M-%S).log"

bash run_with_docker.sh 2>&1 | tee $LOGFILE 
