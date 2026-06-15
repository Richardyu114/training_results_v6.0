export MAX_STEPS=18
export SKIP_EVALS=1
export VAL_CHECK_INTERVAL=4
export LOAD_CKPT=False
export NVTX_FLAG=1
export NSYS_PREFIX="/results/"
export LAYER_CUDA_GRAPH=0
export MEMORY_PROFILE=0
export MCORE_CUDA_GRAPH=True
export CG_WEIGHT_CACHING=False
export NSYS_OUT=/resuts/lora_${SLURM_PROCID}
export NSYS="nsys profile --sample=cpu --cuda-graph-trace=node --cpuctxsw=none --trace=cuda,nvtx -f true --stats true -o ${NSYS_OUT}"
