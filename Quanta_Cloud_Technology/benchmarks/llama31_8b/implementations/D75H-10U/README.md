## Steps to launch training

### QuantaGrid D75H-10U

Launch configuration and system-specific hyperparameters for the QuantaGrid D75H-10U
submission are in the `../<implementation>/D75H-10U_nemo/config_D75H-10U_1x8x2xtp1pp1cp1_8b_fp4.sh` script.

Steps required to launch training on QuantaGrid D75H-10U.

1. Build the docker container and push to a docker registry

```
cd ../D75H-10U_nemo
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
export LOGDIR=<path/to/results_log>
export CONT="<path_to_enroot_image_name>.sqsh"
source config_D75H-10U_1x8x2xtp1pp1cp1_8b_fp4.sh
NEXP=10 sbatch --gpus=${DGXNGPU} -N ${DGXNNODES} run.sub
