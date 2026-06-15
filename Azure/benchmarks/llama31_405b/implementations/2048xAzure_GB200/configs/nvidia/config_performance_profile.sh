export BREAKDOWN=0 # Set to 1 to enable performance breakdown. Also adds NVTX ranges in NeMo and MCore.

# Nsys profiling
export PROFILE=1
export PROFILE_START_STEP=4
export PROFILE_END_STEP=8
export NVTX_FLAG=1
export LIMIT_TRAIN_BATCHES=8
export MAX_STEPS=8
export LIMIT_VAL_BATCHES=2
export VAL_CHECK_INTERVAL=4
export PROFILE_RANKS=0,1,2,3,4,5,6,7
export WALLTIME_RUNANDTIME=20
export WALLTIME=30

# Performance breakdown
if [[ "${BREAKDOWN:-0}" == "1" ]]; then
    export DLSIM_REPORT_PATH="/lustre/share/coreai_mlperf_training/dlsim/report.xlsx" # Point to DLSim report on host filesystem
    export NVTE_NVTX_ENABLED=1 # Enable NVTX ranges in TE
    export LIMIT_VAL_BATCHES=0 # Disable validation
fi
