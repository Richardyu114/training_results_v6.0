## Running NVIDIA Large MoE DeepSeek v3 671B PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Large MoE DeepSeek v3 671B PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 1.5TB disk space is required.
- NVIDIA GPU with at least 190GB memory is strongly recommended (e.g. GB200)
- GPUs are required for the checkpoint processing.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt
export CONT=<docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

**DeepSeek v3 671B uses the same dataset as LLama3.1 8b!**
* **If you already have the LLama3.1 8b dataset you can skip this part**
* **If you don't, please mind that there are pointers/names related to 8b - this has been done on purpose**

Set the directory for the data to be downloaded to:
```bash
export DATADIR=<path/to/dataset>
export HF_DIR="$DATADIR"/config  # the below script will download a HF model config to `HF_DIR`
```

To download the dataset and align the directories with the layout the benchmark expects, run:

```bash
bash data_scripts/download.sh
```

At the end, the directory structure should look like:

```bash
$tree $DATADIR/
$DATADIR/
|-- LICENSE.txt
|-- NOTICE.txt
|-- c4-train.en_6_text_document.bin
|-- c4-train.en_6_text_document.idx
|-- c4-validation-91205-samples.en_text_document.bin
|-- c4-validation-91205-samples.en_text_document.idx
|-- llama-3-1-8b-preprocessed-c4-dataset.md5
|-- tokenizer
|   |-- LICENSE
|   |-- README.md
|   |-- USE_POLICY.md
|   |-- llama-3-1-8b-tokenizer.md5
|   |-- special_tokens_map.json
|   |-- tokenizer.json
|   `-- tokenizer_config.json
`-- config
    |-- hub
    |   `-- models--deepseek-ai--DeepSeek-V3
    |       |-- blobs
    |       |-- refs
    |       |   `-- main
    |       `-- snapshots
    |           `-- e815299b0bcbac849fa540c768ef21845365c9eb
    |               |-- config.json
    |               `-- configuration_deepseek.py
    `-- modules
        |-- __init__.py
        `-- transformers_modules
            |-- __init__.py
            `-- deepseek_hyphen_ai
                |-- __init__.py
                `-- DeepSeek_hyphen_V3
                    |-- __init__.py
                    `-- e815299b0bcbac849fa540c768ef21845365c9eb
                        |-- __init__.py
                        |-- config.json
                        `-- configuration_deepseek.py
```

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

[Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository uses [NeMo Megatron](https://github.com/NVIDIA/NeMo). NeMo Megatron GPT has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8/FP4 training on NVIDIA GPUs.

#### 3.3.2 Model Architecture

The model largely follows the paper titled [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).

| Parameter | Value                       |
|-----------|-----------------------------|
| Model Size | 671B parameters             |
| Architecture | Mixture of Experts |
| Sequence Length | 4096                        |



#### 3.3.3 Checkpoint download

MLCommons hosts the checkpoint for download **exclusively by MLCommons Members** at [MLCommons storage](https://training.mlcommons-storage.org/index.html#deepseekv3-benchmark).

To download, use the [MLCommons R2 Downloader](https://github.com/mlcommons/r2-downloader):

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/deepseekv3_checkpoint_bf16.uri
```

The checkpoint is ~1.4TB.

#### 3.3.4 Starting checkpoint

The checkpoint distributed for this benchmark is provided in HuggingFace format.

It was produced as follows: the original HuggingFace DeepSeek V3 671B checkpoint was loaded into Megatron-Bridge and trained for **50 iterations** with a sequence auxiliary load balancing loss weight of **1e-2**, then converted back to HuggingFace format. This warm-up step is necessary because this benchmark uses the Llama 3.1 8B tokenizer instead of the original DeepSeek tokenizer -- the change in token distribution causes MoE router load imbalance, and the 50 iterations allow the router to adapt before the main benchmark training begins.

#### 3.3.5 Checkpoint preprocessing

The distributed checkpoint is in HuggingFace format and must be converted to Megatron-LM format before training. Follow [the instructions](https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo#checkpoint-preprocessing) from the reference implementation.

After conversion, set `LOAD_CHECKPOINTS_DIR` in your config file to the path of the converted checkpoint before launching the job.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export DATADIR=<as/set/above>
export CONT=<as/set/above>
source config_GB300_32x4x480xtp1pp4ep32cp1_mxfp8.sh  # select config and source it 
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>xtpXppYcpZ.sh`, where X represents tensor parallel, Y represents pipeline parallel, and Z represents context parallel.

# 5. Quality

### Quality metric
Log Perplexity

### Quality target
3.6

### Evaluation frequency
Evaluate after every one step

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
