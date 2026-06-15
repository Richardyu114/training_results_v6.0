## Steps to launch training

### QuantaGrid D75U-1U (NVIDIA GB300 NVL72)

Launch configuration and system-specific hyperparameters for the QuantaGrid D75U-1U
submission are in the `../<implementation>/D75U-1U_hugectr/config_D75U-1U_2x4x6912.sh` script.

Steps required to launch training on QuantaGrid D75U-1U.

1. Build the docker container and push to a docker registry

```
cd ../D75U-1U_hugectr
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Transfer the docker image to enroot container image

```
enroot import -o <path_to_enroot_image_name>.sqsh dockerd://<docker/registry:benchmark-tag>
```

3. Launch the training
```
export DATADIR=<path/to/data>
export DATADIR_VAL=<path/to/val_data>
export LOGDIR=<path/to/results_log>
export CONT="<path_to_enroot_image_name>.sqsh"
source config_D75U-1U_2x4x6912.sh
NEXP=10 sbatch --gpus=${DGXNGPU} -N ${DGXNNODES} run.sub
