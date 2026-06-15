## Steps to launch training

### theia-cmh_ngpu32_ngc26.03_nemo

Launch configuration and system-specific hyperparameters for the
theia-cmh_ngpu32_ngc26.03_nemo submission are in the
`benchmarks/flux1/implementations/theia-cmh_ngpu32_ngc26.03_nemo/config_GB300_08x04x32.sh` script.

Steps required to launch training for theia-cmh_ngpu32_ngc26.03_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB300_08x04x32.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
