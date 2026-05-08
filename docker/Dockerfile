# PSI-0 Training Docker Image
# Supports: fine-tuning on local datasets (LeRobot v2.1) and HuggingFace datasets
#
# Build:
#   docker build -t psi0-train -f docker/Dockerfile .
#
# Run (local dataset):
#   docker run --gpus all --rm \
#     -v $(pwd)/pick_cylinder_manipulation_psi:/workspace/data/pick_cylinder_manipulation_psi \
#     -v /home/marcos/hfm/cache/checkpoints:/workspace/checkpoints \
#     -e WANDB_API_KEY=<your_key> \
#     psi0-train pick_cylinder_manipulation_psi
#
# Run (HuggingFace dataset, auto-download):
#   docker run --gpus all --rm \
#     -v /home/marcos/hfm/cache/checkpoints:/workspace/checkpoints \
#     -e WANDB_API_KEY=<your_key> \
#     -e HF_TOKEN=<your_token> \
#     psi0-train <hf_dataset_repo_id>

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# ── System dependencies ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git git-lfs \
        ffmpeg \
        libaio-dev \
        build-essential \
        curl wget \
        ca-certificates \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Copy repo (excluding data, runs, wandb, venvs) ────────────────────────────
COPY pyproject.toml /workspace/pyproject.toml
COPY src/psi /workspace/src/psi
COPY scripts /workspace/scripts
COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh
RUN chmod +x /workspace/docker/entrypoint.sh

# ── Python virtual environment ─────────────────────────────────────────────────
RUN uv venv .venv --python python3.10
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"

# ── Install PyTorch (cu126 pre-built wheels) ───────────────────────────────────
RUN uv pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchcodec==0.4.0 \
    triton==3.3.0 \
    --index-url https://download.pytorch.org/whl/cu126

# ── Install flash-attn (pre-built wheel for torch2.7+cu12+cp310+cxx11abiTRUE) ──
RUN uv pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

# ── Install DeepSpeed (no CUDA JIT ops, CPU-only build) ───────────────────────
RUN DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 uv pip install deepspeed==0.17.1

# ── Install remaining PSI dependencies ────────────────────────────────────────
RUN uv pip install \
    transformers==4.57.0 \
    accelerate==1.7.0 \
    peft==0.17.1 \
    qwen-vl-utils==0.0.14 \
    sentencepiece \
    "lerobot @ git+https://github.com/songlin/lerobot.git@09929d8057b044b53aecaf5c6d7eb71f99e8beb9" \
    albumentations==1.4.18 \
    datasets==3.6.0 \
    dm-tree==0.1.8 \
    einops==0.8.1 \
    "einx>=0.3.0" \
    h5py==3.14.0 \
    imageio==2.34.2 \
    jsonlines \
    "numpy<2.0.0" \
    numpydantic==1.6.7 \
    "omegaconf>=2.3.0" \
    opencv-python-headless \
    pydantic==2.10.6 \
    "pydantic-yaml>=1.6.0" \
    tqdm \
    typer \
    tyro==0.9.32 \
    wandb \
    simplejpeg \
    python-dotenv \
    huggingface-hub \
    tensorboard \
    setuptools

# ── Install psi package (editable) ────────────────────────────────────────────
RUN uv pip install -e . --no-deps

# ── Patch deepspeed: silence MissingCUDAException on is_compatible() ──────────
COPY docker/patch_deepspeed.py /workspace/docker/patch_deepspeed.py
RUN python /workspace/docker/patch_deepspeed.py

# ── Verify installation ────────────────────────────────────────────────────────
COPY docker/verify_install.py /workspace/docker/verify_install.py
RUN python /workspace/docker/verify_install.py

# ── Runtime environment ────────────────────────────────────────────────────────
ENV PSI_HOME=/workspace/checkpoints
ENV DATA_HOME=/workspace/data
ENV HF_HOME=/workspace/hf_cache
ENV TORCH_HOME=/workspace/hf_cache
ENV HF_LEROBOT_HOME=/workspace/data/lerobot
ENV OMP_NUM_THREADS=8
ENV TOKENIZERS_PARALLELISM=false
ENV DEEPSPEED_LOG_LEVEL=warning
ENV CUDA_LAUNCH_BLOCKING=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV AV_LOG_LEVEL=quiet
ENV DS_BUILD_OPS=0
ENV DS_SKIP_CUDA_CHECK=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
