"""This script contains loading functions for expert trajectories."""

import numpy as np
import lerobot
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset


class EnrichedRobotDataset(torch.utils.data.Dataset):
    """Enriched dataset that includes subsequent goals of the original expert dataset."""

    def __init__(self, original_dataset, goal_horizon, lerobot_dataset):
        """Initialize the dataset.

        Inputs:
            original_dataset: the original dataset to enrich (obtained from LeRobot for example)
            goal_horizon: the number of steps to look ahead for the goal (must be at least action_horizon - obs_horizon + 1)
        """
        self.original_dataset = original_dataset
        self.enriched_data = []
        self.goal_horizon = goal_horizon
        self.lerobot_dataset = lerobot_dataset
        self._enrich_dataset()

    def _enrich_dataset(self):
        n = len(self.original_dataset)
        for i in range(n):
            item = self.original_dataset[i]
            agent_pos = item["observation.state"]  # (obs_horizon, 2)
            action = item["action"]  # (action_horizon, 2)
            image = item["observation.image"]  # (obs_horizon, C, H, W)

            episode_index = item["episode_index"]
            current_episode_end = self.lerobot_dataset.episode_data_index[
                "to"
            ][episode_index].item()
            action_horizon = action.shape[0]
            obs_horizon = agent_pos.shape[0]
            if self.goal_horizon is None:
                self.goal_horizon = current_episode_end
            assert self.goal_horizon > action_horizon - obs_horizon + 1, (
                f"goal_horizon ({self.goal_horizon}) must be greater than "
                f"action_horizon ({action_horizon}) - obs_horizon ({obs_horizon}) + 1"
            )
            # iteration at most until the end of the current episode
            max_horizon = min(i + self.goal_horizon, current_episode_end)
            for j in range(i + 1, max_horizon):
                # add future goals at least action_horizon - obs_horizon + 1 steps ahead and beyond
                if j > i + action_horizon - obs_horizon + 1:
                    # print(f"Enriching dataset: {i}/{n} - {j}/{n}")
                    future_goal_pos = self.original_dataset[j][
                        "observation.state"
                    ]
                    future_goal_image = self.original_dataset[j][
                        "observation.image"
                    ]
                    self.enriched_data.append(
                        {
                            "agent_pos": agent_pos,
                            "action": action,
                            "image": image,
                            "reached_goal_agent_pos": future_goal_pos,
                            "reached_goal_image": future_goal_image,
                        }
                    )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.enriched_data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return self.enriched_data[idx]


def build_expert_dataset(
    cfg: DictConfig,
    dataset_id: str,
    num_demonstrations: int,
    goal_horizon: int,
    # dataset_statistics: dict
):
    """
    Build a dataset from the given trajectories.

    Inputs:
        cfg : configuration file
        trajectories : list of trajectories
        dataset_statistics : dict containing the statistics of the dataset
    Outputs:
        dataset : torch.utils.data.Dataset
    """
    delta_timestamps = {
        "observation.image": cfg.delta_timestamps.observation_image,
        "observation.state": cfg.delta_timestamps.observation_state,
        "action": cfg.delta_timestamps.action,
    }
    expert_demonstrations = LeRobotDataset(
        dataset_id,
        delta_timestamps,
    )
    num_demonstrations = min(num_demonstrations, len(expert_demonstrations))
    starting_idx = np.random.randint(
        0, len(expert_demonstrations) - num_demonstrations
    )
    rollouts = [
        expert_demonstrations[i]
        for i in range(starting_idx, starting_idx + num_demonstrations)
    ]
    dataset = EnrichedRobotDataset(
        original_dataset=rollouts,
        goal_horizon=goal_horizon,
        lerobot_dataset=expert_demonstrations,
    )
    return dataset
