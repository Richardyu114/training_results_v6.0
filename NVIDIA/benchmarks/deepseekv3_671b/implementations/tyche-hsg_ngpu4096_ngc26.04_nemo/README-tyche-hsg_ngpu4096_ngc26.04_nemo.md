## Steps to launch training

### tyche-hsg_ngpu4096_ngc26.04_nemo

Launch configuration and system-specific hyperparameters for the
tyche-hsg_ngpu4096_ngc26.04_nemo submission are in the
`benchmarks/deepseekv3_671b/implementations/tyche-hsg_ngpu4096_ngc26.04_nemo/config_GB200_1024x4x32xtp2pp4ep32cp1_mxfp8_full_cg.sh` script.

Steps required to launch training for tyche-hsg_ngpu4096_ngc26.04_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB200_1024x4x32xtp2pp4ep32cp1_mxfp8_full_cg.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
