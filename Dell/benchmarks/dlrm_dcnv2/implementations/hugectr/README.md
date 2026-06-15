## Running NVIDIA HugeCTR DLRM DCNv2 MLPerf Benchmark

This file contains the instructions for running the NVIDIA HugeCTR DLRM DCNv2 MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 8TB of fast storage space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:recommendation-hugectr
```

### 3.2 Download dataset

The [original dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) is currently hosted on HuggingFace,
but the files are corrupted, so they can not be used.

The preprocessed dataset can instead be downloaded from [MLCommons Storage](https://training.mlcommons-storage.org/index.html#dlrm-v2-benchmark).

The total size is almost 4TB, so this might take a while.

You can use the provided Slurm script:

```bash
export DATADIR="</path/to/dataset>"  # set your </path/to/dataset>
export CONT=<docker/registry>/mlperf-nvidia:recommendation-hugectr
sbatch -N1 -t 24:00:00 scripts/download_dataset.sub  # you might need to add -A account and -p partition
```

or directly:

```bash
cd $DATADIR
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/dlrmv2-preprocessed-criteo-click-logs.uri
```

Saving the dataset to an empty `data/` directory will yield this final dataset structure:

```
/data/preprocessed_criteo_click_logs
├── LICENSE.txt
├── NOTICE.txt
├── dlrmv2-preprocessed-criteo-click-logs.md5
├── test_data.bin
├── train_data.bin
└── val_data.bin
```

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

To launch training with a Slurm cluster, run:

```bash
# Both DATADIR* env vars should point to the same preprocessed data directory
export DATADIR="</path/to/data/preprocessed_criteo_click_logs>"
export DATADIR_VAL="</path/to/data/preprocessed_criteo_click_logs>"
export LOGDIR="</path/to/output_logdir>"  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:recommendation-hugectr
source config_<system>.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```
