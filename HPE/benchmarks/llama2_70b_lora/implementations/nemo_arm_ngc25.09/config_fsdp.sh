export FSDP=megatron
export DDP_OVERLAP_GRAD_REDUCE=True
export DDP_OVERLAP_PARAM_GATHER=True
export MCORE_CUDA_GRAPH=0
unset CUDA_MAX_CONNECTIONS

#if [[ "${NCCL_UB}" == "1" || "${NCCL_UB}" == "True" || "${NCCL_UB}" == "true" ]]; then
#    export NCCL_CTA_POLICY=1
#fi

