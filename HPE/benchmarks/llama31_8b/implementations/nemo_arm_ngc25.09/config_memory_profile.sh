# This config enables memory profiling. 
# It can be directly sourced, but it's recommended to add a `source config_memory_profile.sh` line after the config that you wish to profile. 
export ENABLE_MEMORY_PROFILER=True

# memory profile file prefix. The resulting profile will be named `<name>.pickle`
export MEMORY_PROFILE_FILENAME="${MEMORY_PROFILE_FILENAME:=""}${DGXSYSTEM:=""}"

# maximum number of entries in the memory profiler. Default to 1e7
export MEMORY_PROFILE_MAX_ENTRIES=1000000

# Whether we do rank 0 profiling only
export MEMORY_PROFILE_RANK_0_ONLY=True

# Memory profiler start location
export MEMORY_PROFILE_START_LOCATION="init"

# memory profiler end location
export MEMORY_PROFILE_END_LOCATION="train_start"

# Whether we force trigger OOM before we stop
# this will request 10x10GB memory on GPU which will likely trigger OOM. 
export FORCE_OOM_BEFORE_STOP=False

# Whether we might encounter OOM during memory profiling
# this will `try` and `except` all errors during training, and dump the tracebacks & memory profiles after we encounter errors. 
export MEMORY_PROFILE_POSSIBLE_OOM=True

# Additionally, we can add custom knobs here to overwrite the training process. e.g.: 
export LIMIT_TRAIN_BATCHES=1
export LIMIT_VAL_BATCHES=1
export VAL_CHECK_INTERVAL=1
export MAX_STEPS=1
