
# vars that should be set by the launcher (Pytorch)
: "${RANK:?RANK not set}"
: "${LOCAL_RANK:?LOCAL_RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${LOCAL_WORLD_SIZE:?LOCAL_WORLD_SIZE not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"
: "${NSYS_METRICS:=0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
readonly node_rank="$((RANK/8))"  # hard-coded ..
readonly NODE_RANK=$(( RANK / LOCAL_WORLD_SIZE ))

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [[ "$RANK" -eq 0 ]]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

NSYS_OUT="/results/${NSYS_PREFIX:="lora"}_${SLURM_JOBID}_n${NODE_RANK}_p${LOCAL_RANK}"
NSYSCMD=""
if [ "${NVTX_FLAG:-0}" -eq 1 ]
then
    NSYSCMD="nsys profile --sample=cpu --cuda-graph-trace=node --cpuctxsw=none --trace=cuda,nvtx -f true --stats true -o ${NSYS_OUT}"
    if [ ${NSYS_METRICS} -gt 0 ]; then
        NSYSCMD="${NSYSCMD} --gpu-metrics-devices=${LOCAL_RANK} --gpu-metrics-set=${NSYS_METRICS_SET} --gpu-metrics-frequency=50000"
    fi
fi

declare -a CMD
if [[ ${LOCAL_WORLD_SIZE} -gt 1 ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( ${NSYSCMD} 'python' '-u')
else
    # interactive run on single node, no need to bind
    CMD=( ${NSYSCMD} 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi
if [ "${node_rank:-0}" -eq 0 ] && [ "${local_rank:-0}" -eq 0 ]
then
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
fi

${LOGGER:-} ${BINDCMD:-} ${CMD[@]} train.py; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
then
    # end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"
    # report result
    result=$(( $end - $start ))
    result_name="LLM_FINETUNING"
    echo "RESULT,$result_name,$result,nvidia,$start_fmt"
fi

