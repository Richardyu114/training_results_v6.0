## Steps to launch training

### theia-cmh_ngpu2048_ngc26.04_nemo

Launch configuration and system-specific hyperparameters for the
theia-cmh_ngpu2048_ngc26.04_nemo submission are in the
`benchmarks/llama31_405b/implementations/theia-cmh_ngpu2048_ngc26.04_nemo/config_GB300_512x4x16xtp2pp8cp2_cg_fp4.sh` script.

Steps required to launch training for theia-cmh_ngpu2048_ngc26.04_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB300_512x4x16xtp2pp8cp2_cg_fp4.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
