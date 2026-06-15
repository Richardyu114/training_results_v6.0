## Running NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 300GB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPU is not needed for preprocessing scripts, but is needed for training.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
export CONT=<docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Download dataset and model + preprocessing

This benchmark uses the [GovReport](https://gov-report-data.github.io/) dataset.

The dataset download/preprocessing scripts are included in the container. To invoke them, you need either a docker or slurm/enroot environment. Start the container, replacing `</path/to/dataset>` with the existing path to where you want to save the dataset and the model weights/tokenizer:

```bash
# docker run -it --rm --network=host --ipc=host --volume </path/to/dataset>:/data $CONT
docker run -it --rm --network=host --ipc=host --volume /mnt/data/mlperf_training/llama2_70b_lora:/data $CONT

# now you should be inside the container in the /workspace/ft-llm directory
python scripts/download_dataset.py --data_dir /data/gov_report  # download and preprocess dataset; takes less than 1 minute
python scripts/download_model.py --model_dir /data/model  # download and preprocess model checkpoint used for initialization; could take up to 30 minutes
```

After both scripts finish you should see the following files in the `/data` directory:

```
/mnt/data/mlperf_training/llama2_70b_lora/
├── gov_report
│   ├── data
│   ├── train.npy
│   └── validation.npy
└── model
    ├── iter_0000000
    │   ├── __0_0.distcp
    │   ├── __0_1.distcp
    │   ├── common.pt
    │   ├── metadata.json
    │   ├── modelopt_run_config.yaml
    │   ├── run_config.yaml
    │   ├── tokenizer
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer_config.json
    │   │   ├── tokenizer.json
    │   │   └── tokenizer.model
    │   └── train_state.pt
    ├── latest_checkpointed_iteration.txt
    └── latest_train_state.pt
```

Exit the container.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure: (customized for Inventec's internal lab environment)

```bash
export CONT=<docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
export DATADIR=/mnt/data/mlperf_training/llama2_70b_lora/gov_report
export MODEL=/mnt/data/mlperf_training/llama2_70b_lora/model
export LOGDIR=../../../../results/P9000IG7_ngc26.04_nemo/llama2_70b_lora
source config_P9000IG7_1x8x1xtp1pp1cp1_fp4.sh
export SLURM_MPI_TYPE=pmi2
sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```

## 5. Evaluation

### Quality metric
Cross entropy loss

### Quality target
0.925

### Evaluation frequency
Every 384 sequences, CEIL(384 / global_batch_size) steps if 384 is not divisible by GBS. Skipping first FLOOR(0.125*global_batch_size+2) evaluations

### Evaluation thoroughness
Evaluation on the validation subset that consists of 173 examples
