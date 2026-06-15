## Steps to launch training

### tyche-hsg_ngpu8_ngc26.04_nemo

Launch configuration and system-specific hyperparameters for the
tyche-hsg_ngpu8_ngc26.04_nemo submission are in the
`benchmarks/gpt_oss_20b/implementations/nemo/config_GB200_2x4x2xtp1pp1cp1ep1_mxfp8.sh` script.

Steps required to launch training for tyche-hsg_ngpu8_ngc26.04_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB200_2x4x2xtp1pp1cp1ep1_mxfp8.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
