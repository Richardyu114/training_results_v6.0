ARG BASE_IMAGE=rocm/pytorch:rocm7.0.2_ubuntu22.04_py3.10_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

# GPU architecture - MI350X = gfx950 (single arch = faster builds)
ARG GPU_ARCH=gfx950

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# -----------------------------------------------------------------------------
# System deps + RDMA userspace (single layer, cleanup included)
# NOTE: libibverbs-dev is REQUIRED for RCCL to compile with IB support
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    cmake \
    pkg-config \
    python3-venv \
    libnuma1 \
    libnuma-dev \
    numactl \
    iproute2 \
    ethtool \
    rdma-core \
    ibverbs-providers \
    libibverbs1 \
    libibverbs-dev \
    librdmacm1 \
    librdmacm-dev \
    ibverbs-utils \
    perftest \
    infiniband-diags \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# -----------------------------------------------------------------------------
# RCCL with InfiniBand support (rebuild from source using install.sh)
# The default RCCL in ROCm images does NOT have IB support compiled in.
# Using rocm-7.0.2 release tag to match host ROCm 7.0.x and base image.
# NOTE: RCCL uses dlopen() for libibverbs, so ldd won't show it - that's normal.
# -----------------------------------------------------------------------------
ARG RCCL_TAG=rocm-7.0.2
ENV RCCL_BUILD_DIR=/tmp/rccl

# Clone RCCL release tag, build with IB support, and install
# Use manual cmake build to pass CMAKE_POLICY_VERSION_MINIMUM (fixes MSCCLPP cmake compat)
# Disable MSCCL++ entirely to avoid cmake version issues and speed up build
RUN git clone https://github.com/ROCm/rccl.git ${RCCL_BUILD_DIR} \
 && cd ${RCCL_BUILD_DIR} \
 && git checkout ${RCCL_TAG} \
 && echo "=== Building RCCL ${RCCL_TAG} for ${GPU_ARCH} with IB support ===" \
 && mkdir -p build && cd build \
 && CXX=/opt/rocm/bin/hipcc cmake .. \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DGPU_TARGETS="${GPU_ARCH}" \
    -DAMDGPU_TARGETS="${GPU_ARCH}" \
    -DCMAKE_HIP_ARCHITECTURES="${GPU_ARCH}" \
    -DBUILD_TESTS=OFF \
    -DENABLE_MSCCLPP=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
 && make -j$(nproc) \
 && make install \
 && echo "=== Verifying RCCL IB support (dlopen-based) ===" \
 && strings /opt/rocm/lib/librccl.so | grep -q "NET/IB" && echo "SUCCESS: IB transport code found!" \
 && rm -rf ${RCCL_BUILD_DIR}

# -----------------------------------------------------------------------------
# Python build tools
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    pybind11 \
    ninja \
    packaging \
    pyYAML \
    cmake==3.28.3

WORKDIR /workspace/deps

# -----------------------------------------------------------------------------
# Megatron-core (clone, install, cleanup .git to save space)
# -----------------------------------------------------------------------------
ARG MEGATRON_COMMIT=bc0be960c2f1fb2c4a57ecd7dec6235c79f85412
RUN git clone --recursive --depth 1 https://github.com/ROCm/Megatron-LM.git megatron_lm \
 && cd megatron_lm && git fetch --depth 1 origin ${MEGATRON_COMMIT} && git checkout ${MEGATRON_COMMIT} \
 && pip uninstall -y megatron-core 2>/dev/null || true \
 && pip install --no-cache-dir -e . \
 && cd megatron/core/datasets && make \
 && find /workspace/deps/megatron_lm -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

ENV PYTHONPATH="${PYTHONPATH}:/workspace/deps/megatron_lm"

# -----------------------------------------------------------------------------
# NeMo (clone, install, cleanup)
# -----------------------------------------------------------------------------
RUN git clone --depth 1 --branch v2.3.0 https://github.com/NVIDIA/NeMo nemo \
 && cd nemo && sed -i '12d' requirements/requirements_nlp.txt \
 && pip install --no-cache-dir --no-build-isolation -e ".[nlp]" \
 && find /workspace/deps/nemo -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# -----------------------------------------------------------------------------
# Python deps (copy only requirements.txt first for better caching)
# -----------------------------------------------------------------------------
COPY requirements.txt /workspace/deps/requirements.txt
RUN pip install --no-cache-dir -r /workspace/deps/requirements.txt

# -----------------------------------------------------------------------------
# Transformer Engine (build from source - wheels didn't work)
# -----------------------------------------------------------------------------
ARG TE_COMMIT=d0ef413b5bfe9ea33b1c8626e99fb46967a60712
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
 && git submodule sync && git submodule update --init --recursive \
 && NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_ROCM_ARCH="${GPU_ARCH}" \
    GPU_ARCHS="${GPU_ARCH}" \
    NVTE_FRAMEWORK='pytorch' \
    NVTE_USE_HIPBLASLT=1 \
    MAX_JOBS=64 \
    PYTORCH_ROCM_ARCH="${GPU_ARCH}" \
    CU_NUM=304 \
    pip install . -v --no-cache-dir --no-build-isolation \
 && rm -rf /workspace/deps/TransformerEngine

# -----------------------------------------------------------------------------
# Final cleanup of pip cache and temp files
# -----------------------------------------------------------------------------
RUN rm -rf ~/.cache/pip /tmp/* /var/tmp/* \
 && find /workspace -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# -----------------------------------------------------------------------------
# Project code - copy only needed files
# Note: .dockerignore excludes slurm logs, results, __pycache__, etc.
# -----------------------------------------------------------------------------
WORKDIR /workspace/code

# Copy source code
COPY src/ ./src/

# Copy configs
COPY configs/ ./configs/
COPY config_*.sh ./

# Copy run scripts
COPY run_and_time.sh run.sub runtime_tunables.sh ./

# -----------------------------------------------------------------------------
# Slurm/pyxis/enroot runtime notes:
#   --container-mounts=/dev/infiniband:/dev/infiniband,/sys/class/infiniband:/sys/class/infiniband:ro
# -----------------------------------------------------------------------------
