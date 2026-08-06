#!/usr/bin/env python3
# Inner launch script for GR00T N1.7 fine-tuning — called via torch.distributed.run.
# Equivalent to src/gr00t/gr00t/experiment/launch_finetune.py but for N1.7.

import os
from pathlib import Path
import json

import tyro

from gr00t.configs.base_config import Config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.experiment.experiment import run


def load_modality_config(modality_config_path: str) -> None:
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def load_checkpoint_model_config(base_model_path: str) -> dict:
    config_path = Path(base_model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    with config_path.open("r") as f:
        return json.load(f)


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [p.strip() for p in ft_config.dataset_path.split(",") if p.strip()]
    if not dataset_paths:
        raise ValueError("dataset_path must contain at least one non-empty path")

    # Build a config with Gr00tN1d7Config as the model type
    config = Config(model=Gr00tN1d7Config())
    config.load_config_path = None

    # Load model hyper-parameters from the checkpoint's config.json
    ckpt_model_cfg = load_checkpoint_model_config(ft_config.base_model_path)
    # Remove keys not in Gr00tN1d7Config (e.g. HF auto-added fields)
    valid_fields = {f.name for f in Gr00tN1d7Config.__dataclass_fields__.values()} \
        if hasattr(Gr00tN1d7Config, "__dataclass_fields__") else set(vars(Gr00tN1d7Config()).keys())
    filtered_cfg = {k: v for k, v in ckpt_model_cfg.items() if k in valid_fields}
    config.model = Gr00tN1d7Config(**filtered_cfg)

    config = config.load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )

    # N1.7 uses shortest_image_edge/crop_fraction (new API).
    # The checkpoint config.json also carries the deprecated image_crop_size/image_target_size
    # fields; having both set simultaneously fails the warn_configs assertion in experiment.py.
    config.model.image_crop_size = None
    config.model.image_target_size = None

    # N1.7 fine-tuning flags (freeze backbone, tune diffusion head by default)
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    config.model.load_bf16 = True  # Load backbone in bf16 to halve backbone VRAM (~6→3 GB)
    config.model.backbone_trainable_params_fp32 = True  # Trainable params (projector) kept fp32 for stability

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.reinit_action_head = ft_config.reinit_action_head
    config.training.optim = "adamw_bnb_8bit"  # 8-bit Adam saves ~8-10 GB optimizer state vs fp32 Adam
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    config.training.eval_strategy = ft_config.eval_strategy
    config.training.eval_steps = ft_config.eval_steps
    config.training.eval_set_split_ratio = ft_config.val_split
    config.training.wandb_project = "finetune-gr00t-n1d7"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.override_pretraining_statistics = ft_config.override_pretraining_statistics

    run(config)
