from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    # from psi.config.data_simple import SimpleDataConfig

import torch
from psi.data.lerobot.compat import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from psi.utils import resolve_path
from psi.config.transform import LerobotRepackTransform

class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """ A wrapper around LeRobotDataset to support multiple datasets.
    """

    def __init__(
        self, 
        data_cfg: LerobotDataConfig, 
        split: str = "train"
    ):
        repo_ids = data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids
        first_repo = repo_ids[0] if isinstance(repo_ids, list) else repo_ids
        dataset_meta = LeRobotDatasetMetadata(first_repo, resolve_path(f"{data_cfg.root_dir}/{first_repo}"))
        assert isinstance(data_cfg.transform.repack, LerobotRepackTransform)
        delta_timestamps = data_cfg.transform.repack.delta_timestamps(dataset_meta.fps)
        episodes = data_cfg.episode_indices(split, dataset_meta.total_episodes)

        if len(repo_ids) > 1:
            root_dir = data_cfg.root_dir
            lerobot_dataset_class = MultiLeRobotDataset
        else:
            repo_ids = first_repo
            root_dir = resolve_path(f"{data_cfg.root_dir}/{first_repo}")
            lerobot_dataset_class = LeRobotDataset

        dataset_kwargs = {
            "root": root_dir,
            "delta_timestamps": delta_timestamps,
            "image_transforms": None,
        }
        if episodes is not None:
            dataset_kwargs["episodes"] = episodes
        self.base_dataset = lerobot_dataset_class(repo_ids, **dataset_kwargs)  # type: ignore
        if episodes is not None and len(self.base_dataset.episode_data_index["from"]) != dataset_meta.total_episodes:
            compact_index = self.base_dataset.episode_data_index
            expanded_index = {
                key: torch.zeros(dataset_meta.total_episodes, dtype=values.dtype)
                for key, values in compact_index.items()
            }
            for compact_episode, original_episode in enumerate(episodes):
                for key in expanded_index:
                    expanded_index[key][original_episode] = compact_index[key][compact_episode]
            self.base_dataset.episode_data_index = expanded_index
        self._cache = {}

    def __getitem__(self, idx) -> dict:
        return self.base_dataset[idx]
    
    def __len__(self):
        return len(self.base_dataset)

    @property
    def episode_data_index(self):
        return self.base_dataset.episode_data_index # type: ignore

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes
    
    @property
    def num_frames(self):
        return self.base_dataset.num_frames
    
    @property
    def meta(self):
        return self.base_dataset.meta # type: ignore

    @property
    def stats(self):
        return self.base_dataset.stats if type(self.base_dataset) == MultiLeRobotDataset else self.base_dataset.meta.stats # type: ignore
