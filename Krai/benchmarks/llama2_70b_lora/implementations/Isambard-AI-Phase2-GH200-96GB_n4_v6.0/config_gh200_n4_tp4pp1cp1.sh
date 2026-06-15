#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_gh200.sh

# hyperparameters
export MINIBS=2
export TP=4
export PP=1
export CP=1

# system parameters
export DGXNNODES=4
