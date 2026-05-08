# GR00T N1.6 training image
# Base: CUDA 12.6 + Ubuntu 22.04 (matches torch==2.7.0+cu126 wheel)
#
# Build:
#   docker build -t gr00t-train src/gr00t/
#
# Train (see baselines/gr00t-n1.6/finetune_gr00t.md for full command):
#   docker run --rm --gpus all \
#     -v /home/marcos/hfm:/hfm \
#     -v /home/marcos/Psi0/src:/workspace/src \
#     -e HF_HOME=/hfm/cache \
#     -e DATASET_PATH=/hfm/data/real/<task> \
#     gr00t-train \
#     python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 \
#       /workspace/src/gr00t/gr00t/experiment/launch_finetune.py <args>

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv \
        curl git ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────────────────────
RUN curl -Lsf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Install Python deps via uv (lock-file reproducible) ───────────────────────
# Venv goes to /opt/venv so it is NOT wiped by the runtime volume mount
# (-v /host/src:/workspace/src would erase /workspace/src/gr00t/.venv)
WORKDIR /build

RUN uv venv /opt/venv --python python3.10

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Copy only what uv needs (better layer caching)
COPY pyproject.toml uv.lock ./

RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project

# ── Install bitsandbytes (8-bit Adam, needed for 24 GB VRAM) ──────────────────
RUN uv pip install bitsandbytes

# ── Runtime environment ───────────────────────────────────────────────────────
# gr00t source is provided at runtime via:  -v /home/marcos/Psi0/src:/workspace/src
# PYTHONPATH makes the package importable without installing it in the image.
ENV PYTHONPATH="/workspace/src:/workspace/src/gr00t"

WORKDIR /workspace

CMD ["bash"]
