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

### 3.1.1 Pull the container

First make sure you are logged onto the NGC docker registry:

```bash
docker login nvcr.io
```

Set the container url:

```bash
export CONT=nvcr.io/nvdlfwea/mlperftv51/llama31_8b-amd:20251003 # for AMD/x86
or
export CONT=nvcr.io/nvdlfwea/mlperftv51/llama31_8b-arm:20251003 # for ARM
```

Download the benchmark container:

```bash
docker pull $CONT
```

### 3.1.2 Extract configs and batch script

The launch of a training job consists of two steps:
- sourcing config files,
- submitting batch script to Slurm.

To obtain config files and batch script you need to extract them from inside the container into your workdir `</path/to/your/workdir>`.

Run the container in an interactive session using `docker run` or `srun`:

```bash
# docker
docker run -it --rm --network=host --ipc=host --volume </path/to/your/workdir>:/mounted $CONT
```

```bash
# slurm
srun --nodes=1 -t 40:00 --pty --container-image=${CONT} --container-mounts=</path/to/your/workdir>:/mounted -p <partition> -A <account> /bin/bash
```

Copy the required files to the mounted directory:

```bash
cp /workspace/llm/config_*.sh /mounted/ && chmod 664 /mounted/config_*.sh  # extract config files and set their permissions
cp /workspace/llm/*.sub /mounted/ && chmod 775 /mounted/run.sub  # extract batch scripts and set their permissions
cp -r /workspace/llm/data_scripts /mounted && chmod 775 /mounted/data_scripts/*  # extract data utils scripts and set their permissions
exit
```

Now you have all files required to launch the training.

### 3.2 Prepare dataset

Set the directory for the data to be downloaded to:
```bash
export DATADIR=<path/to/dataset>
```

To download the dataset and align the directories with the layout the benchmark expects, run:

```bash
bash data_scripts/download_8b.sh
```

At the end, the directory structure should look like:

```bash
$tree 8b/
8b/
|-- LICENSE.txt
|-- NOTICE.txt
|-- c4-train.en_6_text_document.bin
|-- c4-train.en_6_text_document.idx
|-- c4-validation-91205-samples.en_text_document.bin
|-- c4-validation-91205-samples.en_text_document.idx
|-- llama-3-1-8b-preprocessed-c4-dataset.md5
`-- tokenizer
    |-- LICENSE
    |-- README.md
    |-- USE_POLICY.md
    |-- llama-3-1-8b-tokenizer.md5
    |-- special_tokens_map.json
    |-- tokenizer.json
    `-- tokenizer_config.json

2 directories, 14 files
```

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

[Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository uses [NeMo Megatron](https://github.com/NVIDIA/NeMo). NeMo Megatron GPT has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8 training on NVIDIA Hopper GPUs.

#### 3.3.2 List of Layers

The model largely follows the paper titled [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).  

#### 3.3.3 Model checkpoint

The LLama3.1 8B is trained from scratch and is not using a checkpoint.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export DATADIR=<as/set/above>
export CONT=<as/set/above>
source config_GB200_2x4x2xtp1pp1cp1_8b.sh  # select config and source it
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>xtpXppYcpZ.sh`, where X represents tensor parallel, Y represents pipeline parallel, and Z represents context parallel.

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
