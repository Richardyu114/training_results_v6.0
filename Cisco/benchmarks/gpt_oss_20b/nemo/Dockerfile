# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${FROM_IMAGE_NAME}

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

WORKDIR /workspace/

RUN pip install numcodecs==0.16.3
RUN pip install nvidia-mathdx==25.1.1

## CUDNN
RUN NVARCH=$(uname -m |grep x86_64 2>/dev/null || echo sbsa) && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub > /etc/apt/trusted.gpg.d/cuda.asc && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y libcudnn9-cuda-13=9.21.1.3-1 && rm -rf /var/lib/apt/lists/*
ENV CUDNN_VERSION=9.21.1.3

## 1. Megatron-Bridge and Megatron-Core
ARG MBRIDGE_REVISION=4b4b5c94fe0eefdb248fe0faf4c63c22a10a6187
ARG MCORE_REVISION=2436e3df7b6dd4ce4647761553eddc39f4dad55f
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init --recursive && \
    echo MBRIDGE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    cd /workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install -e .

ENV PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"

## 2. Transformer Engine
ARG TE_REVISION=pad_weight_scale_inv
ENV CUSTOM_TE_REVISION ${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/ksivaman/TransformerEngine-1.git transformerengine && \
    cd transformerengine && \
    git submodule init && git submodule update && \
    git checkout ${TE_REVISION} && \
    git remote add vasu https://github.com/vasunvidia/TransformerEngine.git && \
    git fetch vasu vrengasamy/pad_weight_scale_inv && \
    git cherry-pick 7ef11a77aa3945847ed663df06c4897099cb57f5 && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    NVTE_CUDA_ARCHS="90a;100a;103a" NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . \
    ; fi

## 3. HybridEP / DeepEP
# Install nvidia-nvshmem-cu13==3.6.5 wheel so DeepEP finds libnvshmem_host.so + libnvshmem_device.a (base image lacks the .a).
RUN pip install --no-cache-dir nvidia-nvshmem-cu13==3.6.5

# Expose CUDA 13.2 CCCL headers (staged under cccl/) on CPATH so libcu++ includes resolve during the DeepEP build.
ENV CPATH=/usr/local/cuda/include/cccl:${CPATH}

# Build for Hopper (sm_90) and Blackwell (sm_100); both archs are required for cross-arch kernel registration.
RUN git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && git checkout 34152ae28f80bcc3ee38d7a12cb2ad87cfd4ea72 && \
    TORCH_CUDA_ARCH_LIST="9.0;10.0" pip install --no-build-isolation .

## 5. Benchmark dependencies
COPY requirements.txt .
RUN pip uninstall -y transformers torchao && pip install --no-cache-dir -r requirements.txt

## 6. Benchmark code
WORKDIR /workspace/llm
COPY . .
RUN cd /workspace/llm/embedding_lib && pip install --force-reinstall --no-deps --no-build-isolation .
