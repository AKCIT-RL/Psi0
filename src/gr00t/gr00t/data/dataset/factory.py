import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        use_eval = self.config.training.eval_strategy != "no"
        val_split: float = getattr(self.config.training, "val_split", 0.1)

        all_train_datasets = []
        all_eval_datasets = []
        all_train_weights = []
        all_eval_weights = []

        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            train_datasets = []
            eval_datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                common_kwargs = dict(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                )

                if use_eval:
                    # Deterministic episode split: last val_split fraction held out for eval.
                    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
                    ep_loader = LeRobotEpisodeLoader(
                        dataset_path=dataset_path,
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                    )
                    n_episodes = len(ep_loader.episode_lengths)
                    val_split = getattr(self.config.training, "eval_set_split_ratio", 0.1)
                    n_val = max(1, int(n_episodes * val_split))
                    n_train = n_episodes - n_val
                    train_indices = list(range(n_train))
                    val_indices = list(range(n_train, n_episodes))

                    train_ds = ShardedSingleStepDataset(**common_kwargs, episode_indices=train_indices)
                    eval_kwargs = {**common_kwargs, "episode_sampling_rate": 1.0}
                    eval_ds = ShardedSingleStepDataset(**eval_kwargs, episode_indices=val_indices)
                    train_datasets.append(train_ds)
                    eval_datasets.append(eval_ds)
                else:
                    dataset = ShardedSingleStepDataset(**common_kwargs)
                    train_datasets.append(dataset)

            dataset_lengths = np.array([len(ds) for ds in train_datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for ds, relative_length in zip(train_datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_train_datasets.append(ds)
                all_train_weights.append(weight)

            if use_eval:
                eval_lengths = np.array([len(ds) for ds in eval_datasets])
                eval_relative_lengths = eval_lengths / eval_lengths.sum()
                for ds, relative_length in zip(eval_datasets, eval_relative_lengths):
                    weight = relative_length * dataset_spec.mix_ratio
                    all_eval_datasets.append(ds)
                    all_eval_weights.append(weight)

        train_mixture = ShardedMixtureDataset(
            datasets=all_train_datasets,
            weights=all_train_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )

        eval_mixture = None
        if use_eval:
            eval_mixture = ShardedMixtureDataset(
                datasets=all_eval_datasets,
                weights=all_eval_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=False,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )

        return train_mixture, eval_mixture
