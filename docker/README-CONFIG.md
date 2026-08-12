# Docker Build Configuration

This directory contains the Docker setup for PSI-0. The build process is now configurable via `.env.docker`.

## Quick Start

### Using docker-compose (recommended)

1. **Review/edit configuration** (optional):
   ```bash
   cat .env.docker
   # Edit variables as needed
   ```

2. **Build the image**:
   ```bash
   docker compose -f docker/compose.yaml build
   ```

3. **Run container**:
   ```bash
   docker compose -f docker/compose.yaml run --rm psi0 bash
   ```

### Building manually with custom arguments

If you don't want to use docker-compose, pass build args directly:

```bash
docker build \
  --build-arg CUDA_VERSION=13.0.0 \
  --build-arg PYTHON_VERSION=3.10 \
  --build-arg PYTORCH_VERSION=2.7.0 \
  -t psi0-train -f docker/Dockerfile .
```

## Configuration

Edit `.env.docker` to customize:

### CUDA/Ubuntu
- `CUDA_VERSION`: CUDA version (e.g., `13.0.0`, `12.4.1`)
- `CUDNN_VERSION`: cuDNN variant (typically `cudnn` or `cudnn8`)
- `UBUNTU_VERSION`: Ubuntu version (e.g., `ubuntu24.04`, `ubuntu22.04`)

### Python & PyTorch
- `PYTHON_VERSION`: Python version (`3.10`, `3.11`, etc.)
  - ⚠️ **Important**: Pre-built wheels (PyTorch, flash-attn) are specific to Python versions
  - Changing this may require finding compatible wheel versions
  
- `PYTORCH_VERSION`: PyTorch version (e.g., `2.7.0`, `2.6.0`)
- `PYTORCH_TORCHVISION_VERSION`: TorchVision version
- `PYTORCH_TORCHCODEC_VERSION`: TorchCodec version
- `PYTORCH_TRITON_VERSION`: Triton version
- `PYTORCH_CUDA_INDEX`: PyTorch wheel index (`cu126`, `cu124`, `cu121`)

### Dependencies
- `FLASH_ATTN_VERSION`: Flash Attention version
  - ⚠️ **Note**: Wheels include Python version (cp310, cp311, etc.)
  - Available releases: https://github.com/Dao-AILab/flash-attention/releases
  - Current setup assumes cp310 (Python 3.10)
  
- `DEEPSPEED_VERSION`: DeepSpeed version
- `TRANSFORMERS_VERSION`: Transformers library version
- `LEROBOT_VERSION`: LeRobot git commit hash

## Examples

### Use CUDA 12.4 instead of 13.0

```bash
# .env.docker
CUDA_VERSION=12.4.1
CUDNN_VERSION=cudnn8
PYTORCH_CUDA_INDEX=cu124
```

### Upgrade PyTorch to 2.6.0

```bash
# .env.docker
PYTORCH_VERSION=2.6.0
PYTORCH_TORCHVISION_VERSION=0.21.0
PYTORCH_TORCHCODEC_VERSION=0.3.0
PYTORCH_TRITON_VERSION=3.2.0
```

### Important Notes

1. **Wheel compatibility**: Pre-built wheels (PyTorch, flash-attn, etc.) have specific Python/CUDA/platform requirements
   - Check availability before changing versions
   - Flash-attn wheel URL includes Python version (`cp310`, `cp311`, etc.)

2. **Default values**: All variables have fallback defaults in `compose.yaml`, so .env.docker is optional

3. **Environment variables**: You can also override via shell environment:
   ```bash
   CUDA_VERSION=12.4.1 docker compose -f docker/compose.yaml build
   ```