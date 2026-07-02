# BitBot v2 Training Image
#
# Pre-baked with PyTorch 2.12.1+cu132 (Blackwell sm_120 support),
# Stable-Baselines3, Optuna, and all training dependencies.
#
# No runtime pip installs needed. Just mount data and run.
#
# Build:
#   docker build -t ghcr.io/pamelia/improved-doodle/bitbot-train:latest .
#
# Run locally (if you have GPUs):
#   docker run --gpus all -v /path/to/data:/workspace/data ghcr.io/pamelia/improved-doodle/bitbot-train:latest
#
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# PyTorch 2.12.1 with CUDA 13.2 -- supports Blackwell (sm_120)
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132

# RL + ML stack
RUN pip install --no-cache-dir --break-system-packages \
    stable-baselines3==2.9.0 \
    gymnasium==1.3.0 \
    hmmlearn==0.3.3 \
    scikit-learn==1.9.0 \
    optuna==4.9.0

# Data + serialization
RUN pip install --no-cache-dir --break-system-packages \
    pandas==3.0.3 \
    pyarrow==24.0.0 \
    numpy==2.5.0

# ONNX export
RUN pip install --no-cache-dir --break-system-packages \
    onnx==1.22.0 \
    onnxruntime==1.27.0

# Monitoring (optional but useful)
RUN pip install --no-cache-dir --break-system-packages \
    tensorboard==2.21.0

# Verify installation
RUN python -c "\
import torch; \
print(f'PyTorch {torch.__version__}, CUDA support: {torch.version.cuda}'); \
import stable_baselines3; \
print(f'SB3 {stable_baselines3.__version__}'); \
import optuna; \
print(f'Optuna {optuna.__version__}'); \
import gymnasium; \
print(f'Gymnasium {gymnasium.__version__}'); \
import hmmlearn; \
print(f'hmmlearn {hmmlearn.__version__}'); \
import sklearn; \
print(f'scikit-learn {sklearn.__version__}'); \
import pandas; \
print(f'pandas {pandas.__version__}'); \
import pyarrow; \
print(f'pyarrow {pyarrow.__version__}'); \
import numpy; \
print(f'numpy {numpy.__version__}'); \
import onnx; \
print(f'ONNX {onnx.__version__}'); \
import onnxruntime; \
print(f'ONNX Runtime {onnxruntime.__version__}'); \
import tensorboard; \
print(f'TensorBoard {tensorboard.__version__}'); \
print('All imports OK')"

WORKDIR /workspace
