# BitBot v2 Training Image
#
# Pre-baked with PyTorch 2.7.0+cu128 (Blackwell sm_120 support),
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
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

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

# PyTorch 2.7.0 with CUDA 12.8 -- supports Blackwell (sm_120)
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# RL + ML stack
RUN pip install --no-cache-dir --break-system-packages \
    stable-baselines3==2.6.0 \
    gymnasium==1.1.1 \
    hmmlearn==0.3.3 \
    scikit-learn==1.6.1 \
    optuna==4.3.0

# Data + serialization
RUN pip install --no-cache-dir --break-system-packages \
    pandas==2.2.3 \
    pyarrow==19.0.1 \
    numpy==2.2.4

# ONNX export
RUN pip install --no-cache-dir --break-system-packages \
    onnx==1.17.0 \
    onnxruntime==1.21.1

# Monitoring (optional but useful)
RUN pip install --no-cache-dir --break-system-packages \
    tensorboard==2.19.0

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
print('All imports OK')"

WORKDIR /workspace
