## Running NVIDIA Large Language Model Llama 3.1 8B PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Large Language Model Llama 3.1 8B PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 100GB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:llama31_8b-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:llama31_8b-pyt
export CONT=<docker/registry>/mlperf-nvidia:llama31_8b-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

Set the directory for the data to be downloaded to:

```bash
# export DATADIR=<path/to/dataset>
export DATADIR=/mnt/data/mlperf_training/llama31
```

To download the dataset and align the directories with the layout the benchmark expects, run:

```bash
bash data_scripts/download_8b.sh
```

The final content under `${DATADIR}/8b` should look like:

```
/mnt/data/mlperf_training/llama31/8b
├── c4-train.en_6_text_document.bin
├── c4-train.en_6_text_document.idx
├── c4-validation-91205-samples.en_text_document.bin
├── c4-validation-91205-samples.en_text_document.idx
├── LICENSE.txt
├── llama-3-1-8b-preprocessed-c4-dataset.md5
├── NOTICE.txt
└── tokenizer
    ├── LICENSE
    ├── README.md
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── USE_POLICY.md

2 directories, 13 files
```

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

[Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository uses [NeMo Megatron](https://github.com/NVIDIA/NeMo). NeMo Megatron GPT has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP4/FP8 training on NVIDIA GPUs.

#### 3.3.2 List of Layers

The model largely follows the paper titled [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).  

Please refer to the [Model details section](https://github.com/mlcommons/training/tree/fdc27e8e2bf90d5dd1f1c83b79d81deedf87697a/small_llm_pretraining/nemo#4-model) from the reference for more details. 

#### 3.3.3 Model checkpoint

The Llama 3.1 8B model is trained from scratch and is not using a checkpoint.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
export CONT=<docker/registry>/mlperf-nvidia:llama31_8b-pyt
export LOGDIR=../../../../results/P9000IG7_ngc26.04_nemo/llama31_8b
export DATADIR=/mnt/data/mlperf_training/llama31
source config_P9000IG7_1x8x2xtp1pp1cp1_8b_fp4.sh
export SLURM_MPI_TYPE=pmi2
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub
```

# 5. Quality

### Quality metric
Log Perplexity

### Quality target
3.3

### Evaluation frequency
Evaluate after every 12,288 sequences (=100M tokens)

### Evaluation thoroughness
Evaluation on the validation subset that consists of 1024 sequences (=8.4M tokens).

# 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>xtpXppYcpZ.sh`, where X represents tensor parallel (TP), Y represents pipeline parallel (PP), and Z represents context parallel (CP).

Notice here that: 

```
MP = TP * PP * CP
DP = WS // MP = (NNODES * GPUS_PER_NODE) / (TP * PP * CP)
miniBS = GBS // DP
```

where: 
```
MP = model parallelism
TP = tensor parallelism
PP = pipeline parallelism
DP = data parallelism
WS = world size (number of nodes x number of gpus per node)
GBS = global batch size
```

Note: changing `MICRO_BATCH_SIZE` doesn't affect GBS or any of the above parameters.
Effectively it controls gradient accumulation (`GA = miniBS // microBS`).

Recommendation on adjusting the knobs: 
1. GBS should be divisible by `DP * VP`, where VP represents Virtual Pipelining, controlled by environment variable `INTERLEAVED_PIPELINE`. 
2. Model's number of layers, controlled by `OVERWRITTEN_NUM_LAYERS` knob (with default 126), should be divisible by PP * VP. 
   1. It's also recommended that, if you choose to adjust this knob, then you should export `LOAD_CHECKPOINT=""` to disable checkpoint loading, otherwise you will be loading a checkpoint with more layers to a model with fewer layers, which might cause issues. 


### Seeds
NeMo produces dataset index shuffling only on one process and holds the `SEED` value in the file name.
Thus, all processes need to have the same value of `SEED` otherwise will not be able to read the data.
The `SEED` environment variable can be set prior to launching the job, otherwise it is set in `run.sub`.
