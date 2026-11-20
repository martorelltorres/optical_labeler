# TORCH_VARIANT selects the PyTorch wheels: "cpu", or a CUDA tag such as "cu121".
# CUDA wheels bundle the CUDA runtime; the host only needs the NVIDIA driver and the
# NVIDIA Container Toolkit.
FROM python:3.10-slim

ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.3.1
ARG TORCHVISION_VERSION=0.18.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/workspace/.cache/torch

# torch and torchvision are pinned as constraints so that later requirements cannot
# replace them with the default (CUDA) builds from PyPI.
RUN pip install torch==${TORCH_VERSION}+${TORCH_VARIANT} \
        torchvision==${TORCHVISION_VERSION}+${TORCH_VARIANT} \
        --extra-index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip freeze | grep -E '^torch(vision)?==' > /tmp/torch-constraints.txt

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -c /tmp/torch-constraints.txt

WORKDIR /workspace
ENV PYTHONPATH=/workspace
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-t", "."]
