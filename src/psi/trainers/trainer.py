from __future__ import annotations
from typing import Union, Optional, List, Any, TYPE_CHECKING
from pathlib import Path
import torch
import math
from copy import copy
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    from psi.config.config import LaunchConfig, TrainConfig, LoggingConfig
    from psi.config.config import  DataConfig
from accelerate import Accelerator
from transformers.trainer_utils import PredictionOutput
from transformers.optimization import get_scheduler
from torch.optim import Optimizer
from psi.utils import snake_to_pascal
import os
import re
import importlib
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import datetime
import accelerate
import random
import shutil
import json
import statistics

from psi.utils import initialize_overwatch
overwatch = initialize_overwatch(__name__)

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
        self.history: list[float] = []

    def update(self, value: float, global_step: int) -> tuple[bool, bool, float]:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_value": self.best_value,
            "counter": self.counter,
            "history": self.history,
        }

    def load_dict(self, state: dict[str, Any]) -> None:
        self.best_value = float(state["best_value"])
        self.counter = int(state["counter"])
        self.history = [float(value) for value in state["history"]][-self.smooth_window:]

def worker_init_fn(worker_id):
    # print(f"worker_init_fn called by worker {worker_id}")
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class Trainer(ABC):
    cfg: LaunchConfig
    model: Any
    optimizer: torch.optim.Optimizer 
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

    train_dataset: Dataset
    val_dataset: Optional[Dataset]

    def __init__(self, cfg: LaunchConfig, device: Union[torch.device, int]):
        self.device = torch.device(device)
        self.cfg = cfg

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        if cfg.train.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif cfg.train.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        # avoid duplicate run names in a real training run
        # should read the timestamp from command line args instead
        # self.timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
        self.timestamp = self.cfg.timestamp
        self.early_stopping_state = EarlyStoppingState(
            patience=cfg.train.early_stopping_patience,
            smooth_window=cfg.train.early_stopping_smooth_window,
            min_steps=cfg.train.early_stopping_min_steps,
        )

    @property
    def default_early_stopping_metric(self) -> str:
        return "loss"

    @property
    def early_stopping_metric(self) -> str:
        metric = self.cfg.train.early_stopping_metric
        return self.default_early_stopping_metric if metric == "auto" else metric

    @classmethod
    def instantiate(
        cls, cfg: LaunchConfig, device: Union[torch.device, int]
    ) -> "Trainer":
        trainer_name = cfg.train.name
        try:
            parts = re.split(r"[-_]", trainer_name)
            trainer_name = "_".join([p.lower() for p in parts])
            module = importlib.import_module(f"psi.trainers.{trainer_name}")
            trainer_clazz = getattr(module, f"{snake_to_pascal(trainer_name)}Trainer")
        except Exception as e:
            raise ValueError(
                f"fail to import {trainer_name} from psi.trainers"
            ) from e

        return trainer_clazz(cfg, device)

    @property
    def train_cfg(self) -> TrainConfig:
        return self.cfg.train

    @property
    def log_cfg(self) -> LoggingConfig:
        return self.cfg.log

    @property
    def data_cfg(self) -> DataConfig:
        return self.cfg.data

    @property
    def hf_token(self):
        return self.cfg.train.hf_token.read_text().strip() \
            if isinstance(self.cfg.train.hf_token, Path) \
            else os.environ[self.cfg.train.hf_token]

    def get_fsdp_plugin(self) -> accelerate.utils.FullyShardedDataParallelPlugin | None: ...

    def create_datasets(self) -> tuple[Dataset, Dataset|None]: 
        ...

    def create_dataloaders(
        self, train_dataset, val_dataset
    ) -> tuple[DataLoader, DataLoader|None]: ...

    def create_optimizer(self):
        optimizer_kwargs = dict(self.cfg.train.lr_scheduler_kwargs)
        if self.cfg.train.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = self.cfg.train.optimizer_foreach
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            **optimizer_kwargs, # type: ignore
        )

    def create_scheduler(
        self, num_training_steps: int | None = None, optimizer: Optimizer | None = None
    ):
        if num_training_steps is None:
            num_training_steps = self.max_training_steps

        # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            self.lr_scheduler = get_scheduler(
                name=self.cfg.train.lr_scheduler_type,
                optimizer=optimizer if optimizer is not None else self.optimizer,
                num_warmup_steps=self.num_warmup_steps * self.world_size,
                num_training_steps=num_training_steps * self.world_size,
                scheduler_specific_kwargs=self.cfg.train.scheduler_specific_kwargs,
            )
        else:
            self.lr_scheduler = accelerate.utils.DummyScheduler(
                optimizer, total_num_steps=self.max_training_steps, warmup_num_steps=self.num_warmup_steps
            ) # type: ignore

        return self.lr_scheduler

    def log(
        self, metrics: dict[str, Any], start_time: Optional[float] = None
    ) -> None: 
        wandb_dict = copy(metrics)
        for k,v in list(wandb_dict.items()):
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                wandb_dict[k] = v.item()
        # wandb_dict.update({"train/grad_norm": grad_norm})
        # wandb_dict.update(self.get_log_kv())
        self.accelerator.log(wandb_dict, step=self.global_step) # WANDB logging

    def log_validation(self, metrics: dict[str, Any]) -> None:
        self.log(metrics)

    def create_optimizer_and_scheduler(self, num_training_steps: int | None = None):
        optimizer = self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )
    def compute_loss(
        self, batch#, return_outputs=False#, num_items_in_batch=None
    ) -> dict: ...

    def training_step(
        self,
        # model: nn.Module,
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> tuple[bool, dict[str, Any]]: ...

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]: ...

    def evaluate(
        self,
        # eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # ignore_keys: Optional[list[str]] = None,
        # metric_key_prefix: str = "eval",
    ) -> dict[str, float] | None: 
        ...

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput: ...

    @property
    def world_size(self) -> int:
        """Returns the number of processes in the current distributed group."""
        return overwatch.world_size()

    @property
    def device_train_batch_size(self) -> int:
        """Returns the training batch size per process."""
        return self.cfg.train.train_batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        """Returns the number of gradient accumulation steps."""
        return self.cfg.train.gradient_accumulation_steps

    @property
    def global_train_batch_size(self) -> int:
        """Returns the global training batch size."""
        return (
            self.device_train_batch_size
            * self.world_size
            * self.gradient_accumulation_steps
        )

    @property
    def len_train_dataset(self) -> int:
        """Returns the length of the training dataset."""
        if hasattr(self, "_len_train_dataset") and self._len_train_dataset: # type: ignore
            return self._len_train_dataset # type: ignore
        return len(self.train_dataset) # type: ignore

    @len_train_dataset.setter
    def len_train_dataset(self, length: int):
        self._len_train_dataset = length

    @property
    def len_val_dataset(self) -> int:
        """Returns the length of the training dataset."""
        if hasattr(self, "_len_train_dataset") and self._len_val_dataset: # type: ignore
            return self._len_val_dataset # type: ignore
        return len(self.val_dataset) # type: ignore
    
    @len_val_dataset.setter
    def len_val_dataset(self, length: int):
        self._len_val_dataset = length

    @property
    def len_train_dataloader(self) -> int:
        total = self.len_train_dataset / (
            self.device_train_batch_size * self.world_size
        )
        if hasattr(self, "train_dataloader_drop_last") and self.train_dataloader_drop_last: # type: ignore
            return math.floor(total)  # drop the last incomplete batch
        else:
            return math.ceil(total)  # include the last incomplete batch
        

    @property
    def len_val_dataloader(self) -> int:
        if not hasattr(self, "val_dataloader") or self.val_dataloader is None:
            return 0
        total = self.len_val_dataset / (
            self.device_train_batch_size * self.world_size
        )
        if hasattr(self, "val_dataloader_drop_last") and self.val_dataloader_drop_last: # type: ignore
            return math.floor(total)  # drop the last incomplete batch
        else:
            return math.ceil(total)  # include the last incomplete batch

    @property
    def num_steps_per_epoch(self) -> int:
        """number of global steps (sync gradients) per epoch"""
        # assert self.len_train_dataloader > self.gradient_accumulation_steps
        return max(self.len_train_dataloader // self.gradient_accumulation_steps, 1)

    @property
    def max_training_steps(self) -> int:
        """Returns the maximum number of training steps."""
        if self.cfg.train.max_training_steps is not None:
            return self.cfg.train.max_training_steps
        else:
            assert self.cfg.train.num_train_epochs is not None
            return self.num_steps_per_epoch * self.world_size * self.cfg.train.num_train_epochs

    @property
    def max_training_epochs(self) -> int:
        return math.ceil(self.max_training_steps / self.num_steps_per_epoch)

    @property
    def num_warmup_steps(self) -> int:
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.cfg.train.warmup_steps
            if self.cfg.train.warmup_steps is not None
            else math.ceil(self.max_training_steps * self.cfg.train.warmup_ratio)  # type: ignore
        )
        return warmup_steps # because accelerate divides the steps by world_size when preparing the scheduler

    @property
    @abstractmethod
    def task_run_name(self) -> str:
        """Returns the task-specific run name."""

    @property
    def run_name(self) -> str:
        run_name = (
            f"{self.cfg.exp}{self.task_run_name}"
            f".b{self.global_train_batch_size}.gpus{overwatch.world_size()}"
        )

        run_name = f"{run_name}.{self.timestamp}"
        if self.cfg.debug:
            run_name = f"debug-{run_name}"
        return run_name

    @property
    def project_dir(self) -> str:
        """Returns the project directory for saving checkpoints and logs."""
        return os.path.join(
            self.cfg.train.output_dir, self.cfg.train.name, self.run_name
        )

    def next_epoch(self, epoch):
        """called between epochs, e.g. for resetting the distributed samplers"""
        if hasattr(self, "train_sampler") and self.train_sampler is not None: # type: ignore
            self.train_sampler.set_epoch(epoch) # type: ignore

    @abstractmethod
    def init_models(self):
        """Initialize the models for training."""

    # @abstractmethod
    def set_train(self):
        self.model.train()

    # @abstractmethod
    def set_eval(self):
        self.model.eval()

    # @abstractmethod
    # def create_optimizers(self):
    #     ...

    # @abstractmethod
    # def create_lr_schedulers(self):
    #     ...

    # @abstractmethod
    # def train_one_step(self, batch_input, global_step, local_step, accelerator=None):
    #     ...

    def step(self, batch_input, global_step, local_step) -> tuple[bool, dict[str, Any]]:
        """ Perform a single training step. """

        self.local_step = local_step
        self.global_step = global_step

        sync_gradients, losses = self.training_step(batch_input)
        return sync_gradients, losses

    @abstractmethod
    def prepare(self, accelerator: Accelerator) -> DataLoader:
        self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        self.train_dataloader = accelerator.prepare(self.train_dataloader)

        if self.cfg.train.overfit_single_batch:
            overwatch.warning("Overfitting a single batch: reusing first batch every step. set cfg.data.image_aug = False for true memorization.")
            first_batch = next(iter(self.train_dataloader))
            class SingleBatchLoader:
                def __iter__(self): 
                    while True:
                        yield first_batch
                def __len__(self):
                    return 1
            self.train_dataloader = SingleBatchLoader()


        # FIXME if self.train_dataloader.dataset is IterableDataset:
        # assert (
        #     abs(len(self.train_dataloader) - self.len_train_dataloader)
        #     <= 1  # because of drop_last option is lost in prepare.
        # ), f"check calculations again, {len(self.train_dataloader)} != {self.len_train_dataloader}"

        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None: # not using if self.val_dataloader to avoid DataLoader.__len__() being called on iterable dataset
            self.val_dataloader = accelerator.prepare(self.val_dataloader)
        """ NOTE:
            We have to manually record how many steps we have accumulated gradients
            because the accelerator does not do it for us.
        """
        self.accelerator = accelerator
        return self.train_dataloader # type: ignore

    def _save_early_stopping_state(self, checkpoint_dir: str) -> None:
        if not self.cfg.train.early_stopping or not self.early_stopping_state.history:
            return
        if self.accelerator.is_main_process:
            with open(os.path.join(checkpoint_dir, "early_stopping_state.json"), "w") as state_file:
                json.dump(self.early_stopping_state.to_dict(), state_file, indent=2)

    def _save_checkpoint_extras(self, checkpoint_dir: str) -> None:
        pass

    def save_checkpoint(self, global_step: int) -> str | None:
        save_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dir = os.path.join(save_dir, f"ckpt_{global_step}")

        self.accelerator.save_state(ckpt_dir)
        self.accelerator.wait_for_everyone()
        self._save_early_stopping_state(ckpt_dir)
        self._save_checkpoint_extras(ckpt_dir)

        if self.accelerator.is_main_process:
            # Keep only the latest max_checkpoints_to_keep checkpoints
            max_to_keep = self.train_cfg.max_checkpoints_to_keep or 100
            if max_to_keep is not None and max_to_keep > 0:
                # List all checkpoint directories matching ckpt_*
                ckpt_dirs = [d for d in os.listdir(save_dir) if d.startswith("ckpt_") and os.path.isdir(os.path.join(save_dir, d))]
                # Extract step numbers and sort by step (assume ckpt_{step})
                def extract_step(d):
                    try:
                        return int(d.split("ckpt_")[-1])
                    except Exception:
                        return -1
                ckpt_dirs_sorted = sorted(ckpt_dirs, key=extract_step, reverse=True)
                # Remove older checkpoints if exceeding max_to_keep
                for old_ckpt in ckpt_dirs_sorted[max_to_keep:]:
                    old_ckpt_path = os.path.join(save_dir, old_ckpt)
                    if "0000" in old_ckpt:
                        # force keeping ckpt saved every 10k
                        continue
                    try:
                        shutil.rmtree(old_ckpt_path)
                        overwatch.info(f"Removed old checkpoint: {old_ckpt_path}")
                    except Exception as e:
                        overwatch.warning(f"Failed to remove old checkpoint {old_ckpt_path}: {e}")

        self.accelerator.wait_for_everyone()
        return ckpt_dir

    def save_best_checkpoint(self) -> str:
        best_dir = os.path.join(self.project_dir, "checkpoints", "best")
        if self.accelerator.is_main_process and os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(best_dir)
        self.accelerator.wait_for_everyone()
        self._save_early_stopping_state(best_dir)
        self._save_checkpoint_extras(best_dir)
        self.accelerator.wait_for_everyone()
        return best_dir

    def update_early_stopping(self, metrics: dict[str, float], global_step: int) -> bool:
        metric_name = self.early_stopping_metric
        if metric_name in metrics:
            metric_value = float(metrics[metric_name])
        else:
            loss_name = "loss" if "loss" in metrics else "val/bc_loss"
            metric_value = float(metrics[loss_name])
            overwatch.warning(
                f"Early stopping metric '{metric_name}' was not returned; using '{loss_name}'."
            )

        improved, should_stop, smoothed_value = self.early_stopping_state.update(metric_value, global_step)
        state = self.early_stopping_state
        overwatch.info(
            f"Early stopping '{metric_name}': value={metric_value:.6g}, median={smoothed_value:.6g}, "
            f"best={state.best_value:.6g}, counter={state.counter}/{state.patience}"
        )
        self.accelerator.log(
            {
                "early_stopping/metric": metric_value,
                "early_stopping/median": smoothed_value,
                "early_stopping/best": state.best_value,
                "early_stopping/counter": state.counter,
            },
            step=global_step,
        )
        if improved:
            best_dir = self.save_best_checkpoint()
            overwatch.info(f"Saved new best checkpoint to {best_dir}")
        if should_stop and self.accelerator.is_main_process:
            self.upload_best_to_hf()
        return should_stop

    def upload_best_to_hf(self) -> None:
        """Upload checkpoints/best to HF_BEST_UPLOAD_REPO right after early stopping."""
        if getattr(self, "_best_uploaded", False):
            return
        repo_id = os.environ.get("HF_BEST_UPLOAD_REPO", "")
        best_dir = os.path.join(self.project_dir, "checkpoints", "best")
        if not repo_id:
            overwatch.warning("HF_BEST_UPLOAD_REPO not set; skipping best-checkpoint upload.")
            return
        if not os.path.isdir(best_dir):
            overwatch.warning(f"Best checkpoint not found at {best_dir}; skipping upload.")
            return
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
            run_config = os.path.join(self.project_dir, "run_config.json")
            if os.path.isfile(run_config):
                api.upload_file(path_or_fileobj=run_config, path_in_repo="run_config.json", repo_id=repo_id)
            info = api.upload_folder(
                repo_id=repo_id,
                folder_path=best_dir,
                allow_patterns=["*.safetensors", "*.json"],
                commit_message=(
                    f"early-stopped best: {self.early_stopping_metric}="
                    f"{self.early_stopping_state.best_value:.6g}"
                ),
            )
            overwatch.info(f"Uploaded best checkpoint to {repo_id} @ {info.oid}")
            self._best_uploaded = True
        except Exception as exc:  # upload must never crash training shutdown
            overwatch.error(f"Failed to upload best checkpoint to {repo_id}: {exc}")

    def resume_from_checkpoint(self) -> tuple[int, Optional[str]]:
        """ resume from a checkpoint if specified in the config. 
            the checkpoint path can be either:
            1) full path to a checkpoint folder, e.g.
               .runs/trainer_name/run_name/checkpoints/ckpt_xxxxxx
            3) or .runs/trainer_name/run_name (latest)
        """
        if self.cfg.train.resume_from_checkpoint is None:
            return 0, None

        resume_path = self.cfg.train.resume_from_checkpoint
        if os.path.basename(resume_path).startswith("ckpt_"):
            path = os.path.basename(resume_path)
            load_path = resume_path
        else:
            load_path = None
            if os.path.exists(f"{resume_path}/checkpoints"):
                # Get the most recent checkpoint
                dirs = [
                    directory for directory in os.listdir(f"{resume_path}/checkpoints")
                    if directory.startswith("ckpt_")
                ]
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
            else:
                path = None

        if path is None:
            overwatch.critical(
                f"Checkpoint '{self.cfg.train.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.cfg.train.resume_from_checkpoint = None
            initial_global_step = 0
            load_path = None
        else:
            load_path = load_path or os.path.join(resume_path, "checkpoints", path)
            overwatch.info(f"Resuming from checkpoint {load_path}")
            self.accelerator.load_state(load_path)
            initial_global_step = int(path.split("_")[1]) + 1 # prevent from saving to the same checkpoint again
            early_stopping_path = os.path.join(load_path, "early_stopping_state.json")
            if os.path.exists(early_stopping_path):
                with open(early_stopping_path, "r") as state_file:
                    self.early_stopping_state.load_dict(json.load(state_file))
                overwatch.info(f"Restored early stopping state from {early_stopping_path}")

        return initial_global_step, load_path

    @property
    def lr(self):
        return self.get_lr()
    
    def get_lr(self):
        return self.lr_scheduler.get_last_lr()[0]
    
    def get_total_grad_norm(self, params=None, norm_type=2):
        if "DeepSpeedEngine" in self.model.__class__.__name__:
            grad_norm = self.model.get_global_grad_norm()
            return grad_norm
        else:
            total_norm = 0.0
            if params is None:
                params = self.model.parameters()
            for p in params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                
            total_norm = total_norm ** (1. / norm_type)
            return total_norm

    def unwrap_model(self):
        # Function for unwrapping if model was compiled with `torch.compile`.
        model = self.accelerator.unwrap_model(self.model)
        from diffusers.utils.torch_utils import is_compiled_module

        model = model._orig_mod if is_compiled_module(model) else model
        # songlin: revert original forward which is changed by accelerate.prepare to always return fp32
        # if hasattr(model, "_original_forward"):
        #     model.forward = model._original_forward
        return model
    
    # def log(
    #     self, ):
    #     ...

    # def evaluate(self, global_step: int, accelerator: Accelerator):
    #     ...

    def finalize(self):
        ...
