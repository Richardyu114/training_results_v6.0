# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.03-py3
FROM ${FROM_IMAGE_NAME}
WORKDIR /workspace/flux
COPY requirements.txt .

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID


RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

RUN pip install nvidia-mathdx==25.1.1

ARG NEMO_REVISION=v2.6.2
RUN if [ "${NEMO_REVISION}" != SKIP ]; then \
    git clone https://github.com/NVIDIA/NeMo.git /workspace/NeMo && \
    cd /workspace/NeMo && \
    git checkout ${NEMO_REVISION} && \
    echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /NEMO_COMMIT_HASH.env && \
    sed -i "s/opencc/opencc==1.1.9/g" requirements/requirements_nlp.txt && \
    pip install -e ".[multimodal]" && \
    pip install git+https://github.com/NVIDIA-NeMo/Run.git \
    ; fi

ARG MCORE_REVISION=bf1a5035f1f776b0bded8bffa0a36eeb573a7a8e
RUN if [ "${MCORE_REVISION}" != SKIP ]; then \
    pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM && \
    cd /workspace/Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install . && \
    cd megatron/core/datasets && \
    make \
    ; fi
ENV PYTHONPATH="${PYTHONPATH}:/opt/Megatron-LM"

ARG TE_REVISION=SKIP
ENV CUSTOM_TE_REVISION ${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    git clone https://github.com/NVIDIA/TransformerEngine.git /workspace/transformerengine && \
    cd /workspace/transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule init && git submodule update && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1 NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

RUN pip uninstall -y megatron-energon

# install LDM
RUN pip install -r requirements.txt

# to get rid of bitsandbytes logs
RUN pip uninstall -y bitsandbytes

COPY . .

