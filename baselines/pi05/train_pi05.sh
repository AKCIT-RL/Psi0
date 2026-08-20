#!/bin/bash

export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export OPENPI_GRAD_CHECKPOINTING=${OPENPI_GRAD_CHECKPOINTING:-1}
export NCCL_RAS_ENABLE=${NCCL_RAS_ENABLE:-0}

# Conservative defaults for multi-GPU training inside Apptainer.
# These can avoid NCCL hangs during DDP initialization on some PCIe/container setups.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

source .venv-openpi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> "
    echo "Example: $0 G1WholebodyXMoveBendPickTeleop-v0"
    exit 1
fi

task="$1"

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE src/openpi/train_pytorch.py \
        $task \
        --exp_name=${task} \
        --save_interval=10000 \
        --checkpoint_base_dir=.runs/openpi-05 
        # --resume
        # --no-wandb-enabled \