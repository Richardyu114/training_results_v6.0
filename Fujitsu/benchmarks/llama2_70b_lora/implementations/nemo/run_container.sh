#! /bin/bash
#

#    -u $(id -u):$(id -g) \
docker run -it --rm --gpus all --network=host --ipc=host --volume /mnt/data4/work/llama2-v60:/data \
    llama2_70b_lora-pyt_v60:latest /bin/bash
