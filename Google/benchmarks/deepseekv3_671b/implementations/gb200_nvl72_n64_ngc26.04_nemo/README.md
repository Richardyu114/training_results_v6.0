# DeepSeek-V3 671B on 64× NVIDIA GB200 NVL72 (Google A4X)

## Steps to launch training

### gb200_nvl72_n64_ngc26.04_nemo

Launch configuration and system-specific hyperparameters for the
gb200_nvl72_n64_ngc26.04_nemo submission are in the
`benchmarks/deepseekv3_671b/implementations/gb200_nvl72_n64_ngc26.04_nemo/config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh` script.

Steps required to launch training for gb200_nvl72_n64_ngc26.04_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis/enroot containerization plugin.

1. Build the docker container and convert it to a squashfs image

```
docker build --pull -t <docker/registry:benchmark-tag> .
enroot import -o <path/to/images>/deepseekv3_671b.sqsh dockerd://<docker/registry:benchmark-tag>
```

2. Prepare dataset and checkpoint

Follow the original MLPerf Training v6.0 DeepSeek-V3 671B README for:
- Dataset download (`bash data_scripts/download.sh`)
- Checkpoint download (MLCommons R2 downloader)
- Checkpoint conversion (HF → Megatron-Bridge)

We placed both the preprocessed dataset and the converted Megatron-Bridge
checkpoint on local SSD (`/mnt/localssd`) on each compute node for fastest
load. Lustre or any other shared filesystem also works — just point `DATADIR`
and `LOAD_CHECKPOINTS_DIR` at wherever you put them.

Expected layout (paths shown for our local-SSD setup):
```
/mnt/localssd/dataset/8b/
├── LICENSE.txt
├── NOTICE.txt
├── c4-train.en_6_text_document.bin           (~84 GB)
├── c4-train.en_6_text_document.idx           (~912 MB)
├── c4-validation-91205-samples.en_text_document.bin   (~167 MB)
├── c4-validation-91205-samples.en_text_document.idx   (~1.8 MB)
├── llama-3-1-8b-preprocessed-c4-dataset.md5
└── tokenizer/

/mnt/localssd/mbridge_ckpt/iter_0000000/
├── __*_*.distcp                   # ~1.3 TB total
├── ...
└── metadata.json
```

`benchmarks/deepseekv3_671b/implementations/gb200_megatron_bridge/expected-mounts.csv` contains the full list of our host-to-container mount mappings used by `run.sub`.


3. Launch the training. Run the following commands:
```
source config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh
export CONT=<path/to/images>/deepseekv3_671b.sqsh
export HF_DIR=<path/to/hf_home>
export LOGDIR=<path/to/output/dir>
export DATADIR=/mnt/localssd/dataset <or your path/to/data/dir>
export LOAD_CHECKPOINTS_DIR=/mnt/localssd/mbridge_ckpt <or your path/to/mbridge_ckpt>

export SEED=3   # We used 3, 4, 5 for the three required runs
export MLPERF_SUBMISSION_ORG=
export MLPERF_SUBMISSION_PLATFORM=GB200_NVL72
sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```

## Notes

- **NVLink domain alignment is required.** This submission uses EP=32 with `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32`, so each expert-parallel group of 32 ranks (8 nodes x 4 GPU) must reside in a single NVL72 domain (one rack). Select exactly 16 nodes from each of 4 NVLink domains (cliques); each rack will host 2 EP groups.

- **HybridEP is built for Blackwell.** The provided
  Dockerfile builds DeepEP with `TORCH_CUDA_ARCH_LIST="10.0;10.3"` for GB200 (sm_100) and GB300 (sm_103).
