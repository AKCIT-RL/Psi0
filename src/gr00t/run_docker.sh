#!/bin/bash
set -e

docker run --rm --gpus all \
    --env-file ./gr00t.env \
    -v /home/lucas_olives/hfm:/hfm \
    -v /home/lucas_olives/Documents/ih/Psi0:/workspace \
    psi/gr00t \
    python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 \
        /workspace/baselines/gr00t-n1.7/launch_finetune_n1d7_inner.py \
        --base-model-path /workspace/checkpoints/GR00T-N1.7-3B \
        --dataset-path /workspace/data/simple_teleop_g1/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
        --embodiment-tag G1_LOCO_DOWNSTREAM \
        --modality-config-path /workspace/src/gr00t/gr00t/configs/modality/g1_locomanip_n1d7.py \
        --num-gpus 1 \
        --output-dir /workspace/output/checkpoints/gr00t_n1d7_G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
        --save-steps 5000 \
        --save-total-limit 4 \
        --max-steps 30000 \
        --warmup-ratio 0.05 \
        --weight-decay 1e-05 \
        --learning-rate 0.0001 \
        --global-batch-size 1 \
        --gradient-accumulation-steps 2 \
        --gradient-checkpointing \
        --dataloader-num-workers 2 \
        --eval-strategy steps \
        --eval-steps 1000 \
        --val-split 0.1 \
        --use-wandb \
        --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08