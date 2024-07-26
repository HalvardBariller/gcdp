"""This script contains loading functions for expert trajectories."""

import numpy as np
import lerobot
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset

from gcdp.scripts.utils import get_demonstration_statistics


class EnrichedRobotDataset(torch.utils.data.Dataset):
    """Enriched dataset that includes subsequent goals of the original expert dataset."""

    def __init__(self, dataset, goal_horizon, lerobot_dataset, starting_idx):
        """Initialize the dataset.

        Inputs:
            dataset: the dataset to enrich (obtained from LeRobot, usually sliced from the original expert dataset)
            goal_horizon: the number of steps to look ahead for the goal (must be at least action_horizon - obs_horizon + 1)
            lerobot_dataset: the original expert dataset
            starting_idx: the index in the original expert dataset of the first element in dataset
        """
        self.dataset = dataset
        self.enriched_data = []
        self.goal_horizon = goal_horizon
        self.lerobot_dataset = lerobot_dataset
        self.starting_idx = starting_idx
        self._enrich_dataset()

    def _enrich_dataset(self):
        """Build the transition dataset with future goals.

        Goal horizon is the number of steps to look ahead for the goal.
        `g >= h - o + 1` (an observation cannot serve as goal if it is reached before the end of the action horizon)
        # -----------------------------------------------------------------------------------------------
        # (legend: o = n_obs_steps, h = horizon, g = goal_horizon)
        # |timestep                  | n-o+1 | ..... | n     | ..... | n-o+h | ..... | n-o+g  | n-o+g+1 |
        # |observation is used       | YES   | YES   | YES   | NO    | NO    | NO    | NO     | NO      |
        # |action is generated       | YES   | YES   | YES   | YES   | YES   | NO    | NO     | NO      |
        # |observation used as goal  | NO    | NO    | NO    | YES   | YES   | YES   | YES    | NO      |
        # -----------------------------------------------------------------------------------------------
        Since several rollouts are concatenated, the goal horizon for each element must be at most the end of the current episode:
        `g = min(i + g + 1, end_of_episode)` (where `i` is the current index)
        """
        n = len(self.dataset)
        for i in range(n):
            item = self.dataset[i]
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
            assert self.goal_horizon > action_horizon - obs_horizon, (
                f"goal_horizon ({self.goal_horizon}) must be greater than "
                f"action_horizon ({action_horizon}) - obs_horizon ({obs_horizon})q"
            )
            # iteration at most until the end of the current episode
            max_horizon = min(
                i + self.goal_horizon + 1,
                current_episode_end - self.starting_idx,
            )
            for j in range(i, max_horizon):
                # add future goals at least action_horizon - obs_horizon + 1 steps ahead (and beyond)
                if j > i + action_horizon - obs_horizon:
                    # print(f"Enriching dataset: {i}/{n} - {j}/{n}")
                    future_goal_pos = self.dataset[j]["observation.state"][-1]
                    future_goal_image = self.dataset[j]["observation.image"][
                        -1
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


def normalize_expert_input(
    instance,
    key: str,
    demonstration_statistics: dict,
):
    """Normalize the input data of the expert demonstrations."""
    assert (
        key in demonstration_statistics
    ), f"Key {key} not found in demonstration statistics"
    horizon = instance.shape[0]
    assert instance.shape[1:] == demonstration_statistics[key]["max"].shape
    max_val = demonstration_statistics[key]["max"]
    min_val = demonstration_statistics[key]["min"]
    max_tensor = torch.ones(instance.shape) * max_val
    min_tensor = torch.ones(instance.shape) * min_val
    min_max_scaled = (instance - min_tensor) / (max_tensor - min_tensor + 1e-8)
    return min_max_scaled


def batch_normalize_expert_input(
    batch,
    demonstration_statistics: dict,
):
    """Normalize the input data of the expert demonstrations."""
    # assert key in demonstration_statistics, f"Key {key} not found in demonstration statistics"
    # assert instances.shape[2:] == demonstration_statistics[key]["max"].shape
    for key in batch.keys():
        if "image" in key:
            continue
        elif "agent_pos" in key:
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
        min_max_scaled = (batch[key] - min) / (max - min + 1e-8)
        batch[key] = min_max_scaled
    return batch


def load_expert_dataset(
    cfg: DictConfig,
    dataset_id: str,
):
    """Load the expert dataset."""
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
    expert_dataset: Dataset,
    num_episodes: int,
    goal_horizon: int,
):
    """
    Build a dataset from the given trajectories.

    Inputs:
        cfg : configuration file
        expert_dataset : the dataset of expert demonstrations
        num_episodes : the number of episodes to include in the dataset
        goal_horizon : the number of steps to look ahead for the goal
    Outputs:
        dataset (Dataset): the dataset of goal-conditioned expert demonstrations
    """
    expert_demonstrations = expert_dataset
    num_episodes = min(num_episodes, expert_demonstrations.num_episodes)
    starting_episode = np.random.randint(
        0, expert_demonstrations.num_episodes - num_episodes
    )
    from_idx = expert_demonstrations.episode_data_index["from"][
        starting_episode
    ].item()
    to_idx = expert_demonstrations.episode_data_index["to"][
        starting_episode + num_episodes - 1
    ].item()
    rollouts = [expert_demonstrations[idx] for idx in range(from_idx, to_idx)]
    dataset = EnrichedRobotDataset(
        dataset=rollouts,
        goal_horizon=goal_horizon,
        lerobot_dataset=expert_demonstrations,
        starting_idx=from_idx,
    )
    return dataset


# if __name__ == "__main__":
#     import hydra
#     @hydra.main(version_base="1.2", config_path="../config", config_name="config")
#     def main(cfg):
#         demonstration_statistics = get_demonstration_statistics()
#         print(demonstration_statistics)
#         dataset = build_expert_dataset(
#             cfg,
#             "lerobot/pusht",
#             2,
#             16,
#             demonstration_statistics,
#         )
#         print(dataset)
#     main()
