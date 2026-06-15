# Source this config for apilogs collections jobs to ensure short jobs

export MAX_STEPS=20
export VAL_CHECK_INTERVAL=40  # MAX_STEPS (20) * GBS (2), overwrite in config if necessary
export LIMIT_VAL_BATCHES=10
export NCCL_TEST=0
export LOAD_CKPT=False
export SKIP_EVALS=0
export WARMUP=0

export CHECK_COMPLIANCE=0
export APILOG_PRECISION="e4m3.fp32"
export APILOG_MODEL_NAME="llama2_70b_lora"
export BENCHMARK="llama2_70b_lora"
export FRAMEWORK="pytorch"
