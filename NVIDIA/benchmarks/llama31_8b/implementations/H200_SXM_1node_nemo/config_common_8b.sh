export MODEL_SIZE="8b"
export TARGET_LOG_PPL="3.3"
export LOAD_CHECKPOINT=""
export OVERWRITTEN_NUM_LAYERS=32
export VAL_SAMPLES=1024
# Full run to target is 1200000. Allow an external override (e.g. MAX_STEPS=50 for a smoke test);
# the upstream file hardcodes this, which silently ignores a caller-provided value.
export MAX_STEPS=${MAX_STEPS:-1200000}
# LR cosine decay length follows MAX_STEPS, so a short smoke run keeps a self-consistent schedule.
export OPT_LR_DECAY_STEPS=${MAX_STEPS}

export OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=False
export DEFER_EMBEDDING_WGRAD_COMPUTE=False
export NVTE_NORM_FWD_USE_CUDNN=0
export NVTE_NORM_BWD_USE_CUDNN=0
export ALIGN_PARAM_GATHER=False
unset PYTORCH_CUDA_ALLOC_CONF
export USE_TE_OPS=True
export CE_FUSION_IMPL=te
export FP8_HYBRID=True
export BUCKET_SIZE=768000000

export MODEL_TFLOP_PER_SAMPLE=421.59
export LOG_EVERY_N_STEPS=32
export GC_TRAIN=10000
export GC_VALID=10000
