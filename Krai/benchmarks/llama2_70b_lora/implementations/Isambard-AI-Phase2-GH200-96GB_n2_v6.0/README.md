## Running NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 300GB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPU is not needed for preprocessing scripts, but is needed for training.

## 2. Software Requirements

- [Slurm] (https://slurm.schedmd.com/)
- [Podman](https://podman.io/)

## 3. Set up

### 3.1 Download the container

```bash
docker pull docker.io/krai4ai/mlperf-nvidia:llama2_70b_lora-pyt
```

### 3.2 Download dataset and model

This benchmark uses the [GovReport](https://gov-report-data.github.io/) dataset.

Start the container, replacing `</path/to/dataset>` with the existing path to where you want to save the dataset and the model weights/tokenizer:

```bash
docker run -it --rm --gpus all --network=host --ipc=host --volume </path/to/dataset>:/data <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
# now you should be inside the container in the /workspace/ft-llm directory
python scripts/download_dataset.py --data_dir /data/gov_report  # download dataset
python scripts/download_model.py --model_dir /data/model  # download model checkpoint used for initialization; could take up to 30 minutes
```

### 3.3 Preprocess dataset and model

Continue with the previous docker container running and convert dataset to numpy format:

```bash
python scripts/convert_dataset.py --data_dir /data/gov_report
python scripts/convert_model.py --input_name_or_path=/data/model --output_path=/data/model/llama2-70b.nemo
cd /data/model && find . -type f ! -name 'llama2-70b.nemo' -exec rm -f {} + && tar -xvf llama2-70b.nemo
```

After conversion you should see the following files in the `/data` directory:

```bash
gov_report/
    train.npy
    validation.npy
model/
    <hash>_tokenizer.model
    llama2-70b.nemo
    model_config.yaml
    model_weights
```

Exit the container.

## 4. Launch training

For training, we use Slurm with the MPI support to run the container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
export DATADIR=</path/to/dataset>/gov_report # set correct </path/to/dataset>
export MODEL=</path/to/dataset>/model # set correct </path/to/dataset>
export LOGDIR=</path/to/output/dir> # set the place where the output logs will be saved
export CONT=docker.io/krai4ai/mlperf-nvidia:llama2_70b_lora-pyt
bash run.sh
```