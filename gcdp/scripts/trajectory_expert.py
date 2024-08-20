"""This script contains loading functions for expert trajectories."""

import numpy as np
import lerobot
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset
import tqdm

from gcdp.scripts.common.utils import get_demonstration_statistics
from gcdp.scripts.datasets.expert_datasets import (
    EnrichedEvenlySpacedRobotDataset,
    EnrichedSubsequentRobotDataset,
    EnrichedTerminalRobotDataset,
)


def normalize_expert_input(
    instance,
    demonstration_statistics: dict,
):
    """Normalize the input data of the expert demonstrations."""
    # for key in demonstration_statistics.keys():
    for key in instance.keys():
        if "image" in key:
            continue
        elif "state" in key:
            max_val = demonstration_statistics["observation.state"]["max"]
            min_val = demonstration_statistics["observation.state"]["min"]
        elif "action" in key:
            max_val = demonstration_statistics["action"]["max"]
            min_val = demonstration_statistics["action"]["min"]
        else:
            raise ValueError(
                f"Key {key} not found in demonstration statistics"
            )
        max = torch.ones(instance[key].shape) * max_val
        min = torch.ones(instance[key].shape) * min_val
        # min_max normalization [0, 1]
        min_max_scaled = (instance[key] - min) / (max - min + 1e-8)
        # normalize to [-1, 1]
        instance[key] = min_max_scaled * 2 - 1
    return instance


def batch_normalize_expert_input(
    batch,
    demonstration_statistics: dict,
):
    """Normalize the input data of the expert demonstrations."""
    normalisation_modes = ["observation.state", "action"]
    for key in normalisation_modes:
        if "image" in key:
            continue
        elif "state" in key:
            max_val = demonstration_statistics["observation.state"]["max"]
            min_val = demonstration_statistics["observation.state"]["min"]
        elif "action" in key:
            max_val = demonstration_statistics["action"]["max"]
            min_val = demonstration_statistics["action"]["min"]
        else:
            raise ValueError(
                f"Key {key} not found in demonstration statistics"
            )
        max = torch.ones(batch[key].shape) * max_val
        min = torch.ones(batch[key].shape) * min_val
        # min_max normalization [0, 1]
        min_max_scaled = (batch[key] - min) / (max - min + 1e-8)
        # normalize to [-1, 1]
        batch[key] = min_max_scaled * 2 - 1
    return batch


class NormalizedExpertDataset(torch.utils.data.Dataset):
    """Normalized expert dataset."""

    def __init__(self, expert_dataset, demonstration_statistics):
        """Initialize the dataset."""
        self.expert_dataset = expert_dataset
        self.demonstration_statistics = demonstration_statistics

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.expert_dataset)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        instance = self.expert_dataset[idx]
        return normalize_expert_input(instance, self.demonstration_statistics)


def load_expert_dataset(
    cfg: DictConfig,
    dataset_id: str,
):
    """Load the expert dataset from LeRobot.

    Args:
        cfg: configuration file
        dataset_id: the id of the dataset to load
    Returns:
        expert_demonstrations (LeRobotDataset): the expert demonstrations with configuration timestamps
    """
    delta_timestamps = {
        "observation.image": eval(cfg.delta_timestamps.observation_image),
        "observation.state": eval(cfg.delta_timestamps.observation_state),
        "action": eval(cfg.delta_timestamps.action),
    }
    expert_demonstrations = LeRobotDataset(
        repo_id=dataset_id,
        delta_timestamps=delta_timestamps,
    )
    return expert_demonstrations


def build_expert_dataset(
    cfg: DictConfig,
    expert_demonstrations: Dataset,
    num_episodes: int,
    normalize: bool = True,
):
    """
    Build a dataset from the given trajectories.

    Inputs:
        cfg : configuration file
        expert_dataset : the dataset of expert demonstrations
        num_episodes : the number of episodes to include in the dataset
        normalize : whether to normalize the dataset
    Outputs:
        dataset (Dataset): the dataset of goal-conditioned expert demonstrations
    """
    demonstration_statistics = get_demonstration_statistics()
    num_episodes = min(num_episodes, expert_demonstrations.num_episodes)

    if cfg.expert_data.transitions == "subsequent":
        print(
            "Building dataset with goals immediately subsequent to the current state in the trajectores..."
        )
        starting_episode = np.random.randint(
            0, expert_demonstrations.num_episodes - num_episodes + 1
        )
        from_idx = expert_demonstrations.episode_data_index["from"][
            starting_episode
        ].item()
        to_idx = expert_demonstrations.episode_data_index["to"][
            starting_episode + num_episodes - 1
        ].item()
        rollouts = [
            expert_demonstrations[idx] for idx in range(from_idx, to_idx)
        ]
        num_padding = (
            cfg.model.pred_horizon - cfg.model.obs_horizon + 1
        ) * num_episodes
        # number of steps to look ahead for the goal
        goal_horizon = cfg.expert_data.goal_horizon
        dataset = EnrichedSubsequentRobotDataset(
            dataset=rollouts,
            goal_horizon=goal_horizon,
            lerobot_dataset=expert_demonstrations,
            starting_idx=from_idx,
            num_padding=num_padding,
        )

    elif cfg.expert_data.transitions == "terminal":
        print(
            "Building dataset with goals corresponding to terminal states of the trajectories..."
        )
        episode_indices = np.random.choice(
            expert_demonstrations.num_episodes, num_episodes, replace=False
        )
        episodes_datasets = []
        for episode_index in tqdm.tqdm(episode_indices):
            from_idx = expert_demonstrations.episode_data_index["from"][
                episode_index
            ].item()
            to_idx = expert_demonstrations.episode_data_index["to"][
                episode_index
            ].item()
            rollout = [
                expert_demonstrations[idx] for idx in range(from_idx, to_idx)
            ]
            dataset = EnrichedTerminalRobotDataset(
                dataset=rollout,
            )
            episodes_datasets.append(dataset)
        dataset = torch.utils.data.ConcatDataset(episodes_datasets)

    elif cfg.expert_data.transitions == "evenly_spaced":
        print(
            "Building dataset with goals evenly spaced along the trajectories..."
        )
        episode_indices = np.random.choice(
            expert_demonstrations.num_episodes, num_episodes, replace=False
        )
        episodes_datasets = []
        for episode_index in tqdm.tqdm(episode_indices):
            from_idx = expert_demonstrations.episode_data_index["from"][
                episode_index
            ].item()
            to_idx = expert_demonstrations.episode_data_index["to"][
                episode_index
            ].item()
            rollout = [
                expert_demonstrations[idx] for idx in range(from_idx, to_idx)
            ]
            dataset = EnrichedEvenlySpacedRobotDataset(
                dataset=rollout,
                num_goals=cfg.expert_data.num_goals,
            )
            episodes_datasets.append(dataset)
        dataset = torch.utils.data.ConcatDataset(episodes_datasets)

    else:
        raise ValueError(
            f"Invalid transition type {cfg.expert_data.transitions}"
        )
    if normalize:
        dataset = NormalizedExpertDataset(
            expert_dataset=dataset,
            demonstration_statistics=demonstration_statistics,
        )
    return dataset


if __name__ == "__main__":
    import hydra

    @hydra.main(
        version_base="1.2", config_path="../config", config_name="config"
    )
    def main(cfg):
        """Test script."""
        demonstration_statistics = get_demonstration_statistics()
        expert_dataset = load_expert_dataset(
            cfg,
            "lerobot/pusht",
        )
        dataset = build_expert_dataset(
            cfg,
            expert_dataset,
            3,
            normalize=True,
        )

    main()
