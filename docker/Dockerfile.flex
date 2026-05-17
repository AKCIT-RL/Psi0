# PSI-0 Training Docker Image
# Supports: fine-tuning on local datasets (LeRobot v2.1) and HuggingFace datasets
#
# Build (with docker-compose):
#   docker compose -f docker/compose.yaml build
#   (uses .env.docker for configuration)
#
# Build (manual with custom args):
#   docker build -t psi0-train \
#     --build-arg CUDA_VERSION=13.0.0 \
#     --build-arg PYTHON_VERSION=3.10 \
#     -f docker/Dockerfile .
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

# ── Build arguments (configurable via .env.docker or --build-arg) ─────────────
ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-${UBUNTU_VERSION}

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs \
        ffmpeg \
        libaio-dev \
        build-essential \
        curl wget \
        ca-certificates \
        unzip \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Copy repo (excluding data, runs, wandb, venvs) ────────────────────────────
# COPY pyproject.toml /workspace/pyproject.toml
# COPY src/psi /workspace/src/psi
# COPY scripts /workspace/scripts
# COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh
# RUN chmod +x /workspace/docker/entrypoint.sh
ARG PYTHON_VERSION
RUN uv venv .venv-psi --python ${PYTHON_VERSION} 

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=third_party,target=third_party \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --no-sources --index-strategy unsafe-best-match --active


#     5  uv add huggingface_cli
#     6  uv sync
#    35  uv add torchrun
#    41  uv add accelerate
#    44  uv add torchvision
#    46  uv add diffusers
#    48  uv add qwen_vl_utils
#    54  uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta1/flash_attn4-4.0.0b1-py3-none-any.whl
#    57  uv pip show flash_attn
#    58  uv pip show flash_attn4
#    59  uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl
#    88  uv pip install lerobot @ git+https://github.com/songlin/lerobot.git@09929d8057b044b53aecaf5c6d7eb71f99e8beb9
#    89  uv pip install lerobot@git+https://github.com/songlin/lerobot.git@09929d8057b044b53aecaf5c6d7eb71f99e8beb9
#    90  uv pip install lerobot@git+https://github.com/songlin/lerobot.git@09929d8057b044b53aecaf5c6d7eb71f99e8beb9 --no-deps
#    92  uv pip show av
#    93  uv add av
#    94  uv add torchcodec
#    96  uv pip show torchcodec
#    97  uv add torchcodec==0.12
#    98  uv pip install torchcodec==0.12
#    99  uv pip install torchcodec==0.12 --index-url=https://download.pytorch.org/whl/cpu
#   100  uv pip install -U torchcodec --index-url=https://download.pytorch.org/whl/cu130
#   101  uv pip show torchcodec
#   108  history | grep uv
#   109  uv pip list
#   115  GIT_LFS_SKIP_SMUDGE=1 uv pip install -r baselines/pi05/requirements-openpi.txt
#   124  uv pip list | grep orb
#   125  uv pip install orbax-checkpoint==0.11.12
#   127  uv pip install orbax-checkpoint==0.11.11
#   129  uv pip install orbax-checkpoint==0.11.10
#   131  uv pip install orbax-checkpoint==0.11.9
#   133  uv pip install orbax-checkpoint==0.11.13
#   134  uv pip install -e src/openpi/openpi-client
#   136  uv pip install numpy==2.4.4
#   138  uv pip show jax
#   139  uv pip list | grep jax
#   140  uv pip install jax==0.7.2
#   141  uv pip install jax[cuda13]==0.7.2
#   143  uv pip list | grep jax
#   144  uv pip install jax[cuda12]==0.7.2
#   146  uv pip install jax[cuda13]==0.7.2 jaxtyping==0.2.36
#   149  uv pip install orbax-checkpoint==0.11.18
#   164  history | grep uv
