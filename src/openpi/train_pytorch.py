"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""
from datetime import timedelta

from dotenv import load_dotenv
assert load_dotenv(), "Failed to load .env file. Make sure it exists and contains the necessary environment variables."

import dataclasses
import gc
import json
import logging
import os
import platform
import shutil
import statistics
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    # print("L81", ckpt_dir); exit(0)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            group="openpi_pytorch",
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'

        torch.distributed.init_process_group(backend=backend, init_method="env://",
                                             timeout=timedelta(seconds=7200000), # was 1800000
                                            rank=local_rank,
                                            world_size=world_size
                                            )

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig, episodes: list[int] | None = None):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True, episodes=episodes)
    return data_loader, data_loader.data_config()


# mirrors src/psi/trainers/trainer.py semantics (moving-median smoothing + patience)
class EarlyStoppingState:
    def __init__(self, patience: int, smooth_window: int, min_steps: int):
        if patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if smooth_window < 1:
            raise ValueError("early_stopping_smooth_window must be at least 1")
        self.patience = patience
        self.smooth_window = smooth_window
        self.min_steps = min_steps
        self.best_value = float("inf")
        self.counter = 0
        self.history = []

    def update(self, value: float, global_step: int):
        self.history.append(float(value))
        self.history = self.history[-self.smooth_window:]
        smoothed_value = statistics.median(self.history)
        improved = smoothed_value < self.best_value
        if improved:
            self.best_value = smoothed_value
            self.counter = 0
        else:
            self.counter += 1
        should_stop = self.counter > self.patience and global_step >= self.min_steps
        return improved, should_stop, smoothed_value

    def state_dict(self):
        return {
            "best_value": self.best_value,
            "counter": self.counter,
            "history": self.history,
        }

    def load_state_dict(self, state):
        self.best_value = float(state["best_value"])
        self.counter = int(state["counter"])
        self.history = [float(value) for value in state["history"]][-self.smooth_window:]


def split_episodes(config: _config.TrainConfig):
    """Deterministic episode-level train/val split (same convention as psi0 finetune)."""
    if config.val_episode_fraction <= 0:
        return None, None
    repo_id = config.data.repo_id
    total = _data.LeRobotDatasetMetadata(repo_id).total_episodes
    n_val = max(1, round(total * config.val_episode_fraction))
    perm = np.random.default_rng(config.seed).permutation(total)
    val_eps = sorted(int(e) for e in perm[:n_val])
    train_eps = sorted(int(e) for e in perm[n_val:])
    logging.info(f"Episode split: {len(train_eps)} train / {len(val_eps)} val (of {total})")
    return train_eps, val_eps


# same dimension splits as src/psi/trainers/finetune.py (G1 36-dim action)
_VAL_METRIC_SPLITS = [14, 28, 31, 32, 33, 34, 35]
_VAL_METRIC_LABELS = [
    "err_l1_hand_joints",
    "err_l1_arm_joints",
    "err_l1_torso_rpy",
    "err_l1_height",
    "err_l1_vx",
    "err_l1_vy",
    "err_l1_vyaw",
    "err_l1_target_yaw",
]


def run_validation(model, val_loader, device, config, data_config, use_ddp):
    """Compute val loss and denormalized action L1 error metrics."""
    eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    eval_model.eval()

    norm_stats = data_config.norm_stats["actions"]
    if data_config.use_quantile_norm:
        scale = (np.asarray(norm_stats.q99) - np.asarray(norm_stats.q01)) / 2.0
    else:
        scale = np.asarray(norm_stats.std)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)

    loss_sum = torch.zeros((), device=device)
    err_sum = None
    count = torch.zeros((), device=device)

    with torch.no_grad():
        for observation, actions in val_loader:
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            losses = eval_model(observation, actions)
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            loss_sum += losses.mean().float()

            pred_actions = eval_model.sample_actions(device, observation)
            Tp = min(pred_actions.shape[1], actions.shape[1])
            Da = actions.shape[-1]
            err = (pred_actions[:, :Tp, :Da].float() - actions[:, :Tp]).abs() * scale
            err = err.reshape(-1, Da).mean(dim=0)
            err_sum = err if err_sum is None else err_sum + err
            count += 1

    if use_ddp:
        dist.all_reduce(loss_sum)
        dist.all_reduce(err_sum)
        dist.all_reduce(count)

    avg_loss = (loss_sum / count).item()
    avg_err = (err_sum / count).cpu().numpy()  # (Da,) denormalized per-dim L1
    groups = np.split(avg_err, _VAL_METRIC_SPLITS, axis=-1)
    metrics = {"loss": avg_loss}
    metrics.update({label: float(np.linalg.norm(g)) for label, g in zip(_VAL_METRIC_LABELS, groups)})

    eval_model.train()
    return metrics


def save_best_checkpoint(model, global_step, config, data_config, early_stopping, val_metrics):
    """Save the current model as checkpoints/best (atomic)."""
    tmp_dir = config.checkpoint_dir / "tmp_best"
    final_dir = config.checkpoint_dir / "best"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.save_model(model_to_save, tmp_dir / "model.safetensors")

    norm_stats = data_config.norm_stats
    if norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_dir / "assets" / data_config.asset_id, norm_stats)

    (tmp_dir / "early_stopping_state.json").write_text(
        json.dumps({"global_step": global_step, "val_metrics": val_metrics, **early_stopping.state_dict()}, indent=2)
    )

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info(f"Saved best checkpoint at step {global_step} -> {final_dir}")


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)

        # only keep 5 latetest checkpoints in folder config.checkpoint_dir 
        # except those checkpoints which is multiple of 5000
        existing_checkpoints = sorted(
            [
                int(d.name)
                for d in config.checkpoint_dir.iterdir()
                if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
            ]
        )
        checkpoints_to_keep = set(
            [step for step in existing_checkpoints if step % 5000 == 0]
        )
        checkpoints_to_keep.update(existing_checkpoints[-5:])  # keep latest 5 checkpoints
        for step in existing_checkpoints:
            if step not in checkpoints_to_keep:
                dir_to_remove = config.checkpoint_dir / f"{step}"
                shutil.rmtree(dir_to_remove)
                logging.info(f"Removed old checkpoint at step {step} -> {dir_to_remove}")

def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            # Load with strict=False to allow partial loading and get mismatch info
            missing_keys, unexpected_keys = safetensors.torch.load_model(
                model_to_load, safetensors_path, device=str(device), strict=False
            )
            if missing_keys:
                logging.warning(f"Missing keys when loading checkpoint: {missing_keys[:10]}..." if len(missing_keys) > 10 else f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:10]}..." if len(unexpected_keys) > 10 else f"Unexpected keys: {unexpected_keys}")
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                # raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
                print(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            # raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
            print(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
            
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    train_episodes, val_episodes = split_episodes(config)
    loader, data_config = build_datasets(config, episodes=train_episodes)

    val_loader = None
    if val_episodes is not None:
        val_loader = _data.create_data_loader(
            config,
            framework="pytorch",
            shuffle=False,
            episodes=val_episodes,
            num_batches=config.val_num_batches,
        )

    early_stopping = None
    if config.early_stopping:
        if val_loader is None:
            raise ValueError("early_stopping requires val_episode_fraction > 0")
        early_stopping = EarlyStoppingState(
            patience=config.early_stopping_patience,
            smooth_window=config.early_stopping_smooth_window,
            min_steps=config.early_stopping_min_steps,
        )
        es_state_path = config.checkpoint_dir / "best" / "early_stopping_state.json"
        if resuming and es_state_path.exists():
            early_stopping.load_state_dict(json.loads(es_state_path.read_text()))
            logging.info(f"Resumed early stopping state: best={early_stopping.best_value:.6f} counter={early_stopping.counter}")
    es_metric = config.early_stopping_metric
    if es_metric == "auto":
        es_metric = "err_l1_hand_joints"

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        # Psi-0: adapt action dim to 36
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        pad_dim = config.model.action_dim - state_dict["action_in_proj.weight"].shape[1]
        if pad_dim > 0:
            # eg., torch.Size([1024, 32]) -> torch.Size([1024, 36])
            # Replicate the last 4 columns instead of padding with zeros
            w = state_dict["action_in_proj.weight"]
            to_pad = w[:, -pad_dim:]
            # to_pad = torch.zeros_like(w[:, -pad_dim:])
            state_dict["action_in_proj.weight"] = torch.cat([w, to_pad], dim=1)

            b = state_dict["action_out_proj.bias"]
            # b = torch.zeros_like(state_dict["action_out_proj.bias"])
            state_dict["action_out_proj.bias"] = torch.cat([b, b[-pad_dim:]], dim=0)

            w = state_dict["action_out_proj.weight"]
            to_pad = w[-pad_dim:, :]
            # to_pad = torch.zeros_like(w[-pad_dim:, :])
            state_dict["action_out_proj.weight"] = torch.cat([w, to_pad], dim=0)

        # https://github.com/Physical-Intelligence/openpi/issues/669
        state_dict["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = \
            state_dict["paligemma_with_expert.paligemma.lm_head.weight"]

        _model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        missing_keys, unexpected_keys = _model.load_state_dict(state_dict, strict=False)
        # missing_keys, unexpected_keys = safetensors.torch.load_model(
        #     (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path, strict=False
        # )
        if missing_keys:
            logging.warning(f"Missing keys when loading initial weights: {missing_keys[:10]}..." if len(missing_keys) > 10 else f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in initial weights: {unexpected_keys[:10]}..." if len(unexpected_keys) > 10 else f"Unexpected keys: {unexpected_keys}")
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    countp = lambda m: sum(p.numel() for p in m.values() if p.requires_grad)

    trainables = {}
    for k,v in model.named_parameters():
        # freeze language parts
        # if not "paligemma.model.language_model." in k:

        # freeze vision tower parts
        # if not ".paligemma.model.vision_tower." in k:
        
        # freeze mm projector
        # if not ".paligemma.model.multi_modal_projector." in k:
        
        # freeze paligemma VLM entirely
        if not ".paligemma.model." in k:
            trainables[k] = v

        # always skip lm_head
        if "paligemma_with_expert.gemma_expert.lm_head.weight" in k:
            continue

        # # full finetune
        # trainables[k] = v

    # disable grads for the untrainable parts
    for n,p in model.named_parameters():
        if n not in trainables:
            p.requires_grad = False

    print(f"number of trainable params: {countp(trainables):,} out of {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        trainables.values(), # model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        stop_training = False
        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            #print(loss.item());exit(0)
            if loss.item() > 100:
                logging.warning(f"Abnormally high loss detected at step {global_step}: {loss.item():.4f}")
                torch.save(losses.detach().cpu(), "abnormal_losses.pt")
                torch.save(actions.detach().cpu(), "abnormal_actions.pt")

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )
            if loss.item() > 100:
                logging.warning(f"High loss detected at step {global_step}: {loss.item():.4f}")
                

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                        
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Validation + early stopping (metrics are all-reduced, so every rank
            # takes the same decision)
            if val_loader is not None and global_step % config.val_interval == 0:
                val_metrics = run_validation(model, val_loader, device, config, data_config, use_ddp)
                if is_main:
                    logging.info(
                        "val step=%d %s" % (global_step, " ".join(f"{k}={v:.5f}" for k, v in val_metrics.items()))
                    )
                    if config.wandb_enabled:
                        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
                if early_stopping is not None:
                    improved, should_stop, smoothed = early_stopping.update(val_metrics[es_metric], global_step)
                    if is_main:
                        if config.wandb_enabled:
                            wandb.log(
                                {
                                    f"val/{es_metric}_smoothed": smoothed,
                                    "val/early_stopping_counter": early_stopping.counter,
                                },
                                step=global_step,
                            )
                        if improved:
                            save_best_checkpoint(model, global_step, config, data_config, early_stopping, val_metrics)
                    if should_stop:
                        if is_main:
                            logging.info(
                                f"Early stopping at step {global_step}: {es_metric} did not improve for "
                                f"{early_stopping.counter} validations (best={early_stopping.best_value:.6f})"
                            )
                        stop_training = True

            if stop_training:
                break

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

        if stop_training:
            break

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
