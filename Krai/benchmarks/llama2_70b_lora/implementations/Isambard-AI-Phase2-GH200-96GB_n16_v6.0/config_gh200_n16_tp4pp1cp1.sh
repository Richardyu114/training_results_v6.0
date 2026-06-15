#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_gh200.sh

# hyperparameters
export MINIBS=1
export TP=4
export PP=1
export CP=2

# system parameters
export DGXNNODES=16
