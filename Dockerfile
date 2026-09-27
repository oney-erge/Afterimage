# Afterimage -- lossless streaming inference.
#
# Build for NVIDIA (default):
#   docker build -t afterimage:cuda .
# Build CPU-only (works anywhere, slow -- the GPU decode kernels need CUDA):
#   docker build -t afterimage:cpu --build-arg VARIANT=cpu .
#
# There is no ROCm build target here: this project's ROCm/AMD support has
# not been exercised on real AMD hardware
# and shipping an unverified image would overstate what's actually
# supported. Run --gpus all + docker's ROCm passthrough manually if you
# want to try it.

ARG VARIANT=cuda

# RTX 50-series (Blackwell) hosts need both a newer CUDA runtime and a
# matching torch wheel: rebuild with
#   --build-arg CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04 \
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# Not verified in this environment (no Blackwell/Docker+GPU access here);
# the native-install path's CUDA-version detection is in
# scripts/install-utils.sh's install_cuda_torch_index.
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE} AS base-cuda
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG PIP_EXTRAS=gpu,server

FROM ubuntu:22.04 AS base-cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_EXTRAS=server

FROM base-${VARIANT} AS final
ARG TORCH_INDEX_URL
ARG PIP_EXTRAS

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AFTERIMAGE_STORE_ROOT=/data/stores \
    HF_HOME=/data/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git curl \
    && groupadd --gid 10001 afterimage \
    && useradd --uid 10001 --gid afterimage --create-home afterimage \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY afterimage ./afterimage

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# [gpu] on the CUDA variant declares the triton dependency this image
# actually needs for GPU decode kernels; installing only [server] worked by
# accident (torch's Linux wheel pulls triton in transitively) rather than by
# a declared requirement.
RUN pip install --upgrade pip wheel \
    && pip install torch --index-url ${TORCH_INDEX_URL} \
    && pip install -e ".[${PIP_EXTRAS}]"

RUN mkdir -p /data/stores /data/hf-cache \
    && chown -R afterimage:afterimage /data

VOLUME ["/data"]
EXPOSE 8420

USER afterimage

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8420/health || exit 1

ENTRYPOINT ["afterimage"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8420"]
