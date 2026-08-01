from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Dict, List, TYPE_CHECKING
from psi.config.config import DataConfig
from pathlib import Path
from psi.utils import resolve_data_path
import os
import json
import random

from psi.config.transform import ActionStateTransform
class LerobotDataConfig(DataConfig):
    root_dir: str
    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)
    val_episode_fraction: float | None = None
    val_episodes: List[int] | None = None
    val_episode_seed: int = 42

    @model_validator(mode="after")
    def check_repo_ids(self):
        if len(self.train_repo_ids) == 0:
            raise ValueError("train_repo_ids must be provided")
        if self.val_episode_fraction is not None and self.val_episodes is not None:
            raise ValueError("Only one of val_episode_fraction or val_episodes can be set")
        if self.val_episode_fraction is not None and not 0 < self.val_episode_fraction < 1:
            raise ValueError("val_episode_fraction must be between 0 and 1")
        if (self.val_episode_fraction is not None or self.val_episodes is not None) and len(self.train_repo_ids) != 1:
            raise ValueError("Episode splitting requires exactly one train_repo_id")
        if self.val_episodes is not None and len(self.val_episodes) == 0:
            raise ValueError("val_episodes must not be empty")
        if len(self.val_repo_ids) == 0:
            self.val_repo_ids = [self.train_repo_ids[0]]
        if (self.val_episode_fraction is not None or self.val_episodes is not None) and self.val_repo_ids != self.train_repo_ids:
            raise ValueError("Episode splitting requires train_repo_ids and val_repo_ids to match")
        return self

    def episode_indices(self, split: str, total_episodes: int) -> List[int] | None:
        if self.val_episode_fraction is None and self.val_episodes is None:
            return None

        if self.val_episodes is not None:
            val_episodes = sorted(set(self.val_episodes))
        else:
            num_val_episodes = max(1, round(total_episodes * self.val_episode_fraction))  # type: ignore[arg-type]
            val_episodes = sorted(random.Random(self.val_episode_seed).sample(range(total_episodes), num_val_episodes))

        invalid_episodes = [episode for episode in val_episodes if episode < 0 or episode >= total_episodes]
        if invalid_episodes:
            raise ValueError(f"Validation episodes out of range: {invalid_episodes}")
        if len(val_episodes) >= total_episodes:
            raise ValueError("Episode split must leave at least one training episode")

        print(f"LeRobot validation episodes: {val_episodes}")
        if split == "val":
            return val_episodes
        val_episode_set = set(val_episodes)
        return [episode for episode in range(total_episodes) if episode not in val_episode_set]
    
    @model_validator(mode="after")
    def load_stats(self):
        if not isinstance(self.transform.field, ActionStateTransform):
            return self
        if (
            not Path(self.transform.field.stat_path).is_absolute() and 
            self.transform.field.action_max is None
        ):
            fpath = resolve_data_path(
                Path(self.root_dir) / self.train_repo_ids[0] / self.transform.field.stat_path
            )
            if not os.path.exists(fpath):
                return self
            with open(fpath, "r") as f:
                stats = json.load(f)
                self.transform.field.populate_stats(stats)
        return self

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        from psi.data.lerobot import LeRobotDatasetWrapper
        from psi.data.dataset import Dataset as MapStyleDataset

        data_config = self
        field_transform = self.transform.field
        if (
            split == "val"
            and isinstance(field_transform, ActionStateTransform)
            and (field_transform.state_noise_std > 0.0 or field_transform.state_noise_std_waist > 0.0)
        ):
            data_config = self.model_copy(deep=True)
            data_config.transform.field.state_noise_std = 0.0
            data_config.transform.field.state_noise_std_waist = 0.0

        train_dataset = LeRobotDatasetWrapper(data_config, split=split)
        return MapStyleDataset(data_config, train_dataset, transform_kwargs=transform_kwargs)

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        return dataset[0]
