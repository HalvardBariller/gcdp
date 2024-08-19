"""This script contains loading functions for expert trajectories."""

import numpy as np
import lerobot
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset
import tqdm

from gcdp.scripts.utils import get_demonstration_statistics


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


class EnrichedSubsequentRobotDataset(torch.utils.data.Dataset):
    """Enriched dataset that includes subsequent goals from the original expert dataset."""

    def __init__(
        self, dataset, goal_horizon, lerobot_dataset, starting_idx, num_padding
    ):
        """Initialize the dataset.

        Inputs:
            dataset: the dataset to enrich (obtained from LeRobot, usually sliced from the original expert dataset)
            goal_horizon: the number of steps to look ahead for the goal (must be at least action_horizon - obs_horizon + 1)
            lerobot_dataset: the original expert dataset
            starting_idx: the index in the original expert dataset of the first element in dataset
            num_padding: the number of end-of-sequence padding transitions to add to the enriched dataset
        """
        self.dataset = dataset
        self.enriched_data = []
        self.goal_horizon = goal_horizon
        self.lerobot_dataset = lerobot_dataset
        self.starting_idx = starting_idx
        self.num_padding = num_padding
        self._enrich_dataset()

    def _enrich_dataset(self):
        """Build the transition dataset with future goals.

        Goal horizon is the number of steps to look ahead for the goal.
        `g >= h - o + 1` (an observation cannot serve as goal if it is reached before the end of the action horizon)
        # -----------------------------------------------------------------------------------------------
        # (legend: o = obs_horizon, h = action_horizon, g = goal_horizon)
        # |timestep                  | n-o+1 | ..... | n     | ..... | n-o+h | ..... | n-o+g  | n-o+g+1 |
        # |observation is used       | YES   | YES   | YES   | NO    | NO    | NO    | NO     | NO      |
        # |action is generated       | YES   | YES   | YES   | YES   | YES   | NO    | NO     | NO      |
        # |observation used as goal  | NO    | NO    | NO    | NO    | NO    | YES   | YES    | NO      |
        # -----------------------------------------------------------------------------------------------
        Since several rollouts are concatenated, the goal horizon for each element must be at most the end of the current episode:
        `g = min(i + g + 1, end_of_episode)` (where `i` is the current index)
        """
        n = len(self.dataset)
        cnt_padd = 0
        for i in range(n):
            item = self.dataset[i]
            agent_pos = item["observation.state"]  # (obs_horizon, obs_dim)
            action = item["action"]  # (pred_horizon, action_dim)
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
                    # print(f"Enriching dataset: {i + self.starting_idx}/{current_episode_end} - {j}/{j + self.starting_idx}")
                    future_goal_pos = self.dataset[j]["observation.state"][-1]
                    future_goal_image = self.dataset[j]["observation.image"][
                        -1
                    ]
                    self.enriched_data.append(
                        {
                            "observation.state": agent_pos,
                            "action": action,
                            "observation.image": image,
                            "reached_goal.state": future_goal_pos,
                            "reached_goal.image": future_goal_image,
                        }
                    )
            # add padded transitions to the end of the sequence
            if (
                i + self.goal_horizon + 1
                > current_episode_end - self.starting_idx
            ):
                cnt_padd += 1
                # print(f"Padding dataset: {i}/{n} - {i}/{n}")
                if cnt_padd <= self.num_padding:
                    self.enriched_data.append(
                        {
                            "observation.state": agent_pos,
                            "action": action,
                            "observation.image": image,
                            "reached_goal.state": self.lerobot_dataset[
                                current_episode_end - 1
                            ]["observation.state"][-1],
                            "reached_goal.image": self.lerobot_dataset[
                                current_episode_end - 1
                            ]["observation.image"][-1],
                        }
                    )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.enriched_data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return self.enriched_data[idx]


class EnrichedTerminalRobotDataset(torch.utils.data.Dataset):
    """Enriched dataset that includes terminal goals of the original expert dataset (last achieved position)."""

    def __init__(
        self, dataset, lerobot_dataset, starting_idx, drop_n_last_frames=7
    ):
        """Initialize the dataset.

        Inputs:
            dataset: the dataset to enrich (obtained from LeRobot, usually sliced from the original expert dataset)
            lerobot_dataset: the original expert dataset
            starting_idx: the index in the original expert dataset of the first element in dataset
            drop_n_last_frames: the number of last frames to drop from the enriched dataset
        """
        self.dataset = dataset
        self.enriched_data = []
        self.lerobot_dataset = lerobot_dataset
        self.starting_idx = starting_idx
        self.drop_last = drop_n_last_frames
        self._enrich_dataset()

    def _enrich_dataset(self):
        """Build the transition dataset using the last state of the episode as the goal."""
        n = len(self.dataset)
        for i in range(n):
            item = self.dataset[i]
            agent_pos = item["observation.state"]  # (obs_horizon, obs_dim)
            action = item["action"]  # (pred_horizon, action_dim)
            image = item["observation.image"]  # (obs_horizon, C, H, W)

            episode_index = item["episode_index"]
            current_episode_end = self.lerobot_dataset.episode_data_index[
                "to"
            ][episode_index].item()

            if self.starting_idx + i + self.drop_last < current_episode_end:
                self.enriched_data.append(
                    {
                        "observation.state": agent_pos,
                        "action": action,
                        "observation.image": image,
                        "reached_goal.state": self.lerobot_dataset[
                            current_episode_end - 1
                        ]["observation.state"][-1],
                        "reached_goal.image": self.lerobot_dataset[
                            current_episode_end - 1
                        ]["observation.image"][-1],
                    }
                )
                print(
                    f"Enriching dataset: {i}/{n} - {i + self.starting_idx}/{current_episode_end - 1}"
                )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.enriched_data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return self.enriched_data[idx]


class EnrichedTerminalRobotDataset2(torch.utils.data.Dataset):
    """Enriched dataset of a single rollout using terminal goal of the original expert episode (last achieved position)."""

    def __init__(self, dataset, drop_n_last_frames=7):
        """Initialize the dataset.

        Inputs:
            dataset: a single episode to enrich (obtained from LeRobot original expert dataset)
            drop_n_last_frames: the number of last frames to drop from the episode
        """
        self.rollout = dataset
        self.drop_last = drop_n_last_frames
        self.enriched_data = []
        self._enrich_dataset()

    def _enrich_dataset(self):
        """Build the transition dataset using the last state of the episode as the goal."""
        for i in range(len(self.rollout) - self.drop_last):
            self.enriched_data.append(
                {
                    "observation.state": self.rollout[i][
                        "observation.state"
                    ],  # (obs_horizon, obs_dim)
                    "action": self.rollout[i][
                        "action"
                    ],  # (pred_horizon, action_dim)
                    "observation.image": self.rollout[i][
                        "observation.image"
                    ],  # (obs_horizon, C, H, W)
                    "reached_goal.state": self.rollout[
                        # to_idx - 1
                        -1
                    ]["observation.state"][
                        -1
                    ],  # (obs_dim)
                    "reached_goal.image": self.rollout[
                        # to_idx - 1
                        -1
                    ]["observation.image"][
                        -1
                    ],  # (C, H, W)
                }
            )
            # print(
            #     f"Enriching dataset: {episode_index} | {i}/{len(rollouts)}"
            # )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.enriched_data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return self.enriched_data[idx]


class EnrichedTerminalRobotDataset3(torch.utils.data.Dataset):
    """Enriched dataset that includes subsequent goals from the original expert dataset."""

    def __init__(
        self,
        dataset,
        lerobot_dataset,
        starting_idx,
        drop_n_last_frames=7,
    ):
        """Initialize the dataset.

        Inputs:
            dataset: the dataset to enrich (obtained from LeRobot, usually sliced from the original expert dataset)
            goal_horizon: the number of steps to look ahead for the goal (must be at least action_horizon - obs_horizon + 1)
            lerobot_dataset: the original expert dataset
            starting_idx: the index in the original expert dataset of the first element in dataset
            num_padding: the number of end-of-sequence padding transitions to add to the enriched dataset
        """
        self.dataset = dataset
        self.enriched_data = []
        self.lerobot_dataset = lerobot_dataset
        self.starting_idx = starting_idx
        self.drop_last = drop_n_last_frames
        self._enrich_dataset()

    def _enrich_dataset(self):
        n = len(self.dataset)
        cnt_padd = 0
        for i in range(n):
            item = self.dataset[i]
            agent_pos = item["observation.state"]  # (obs_horizon, obs_dim)
            action = item["action"]  # (pred_horizon, action_dim)
            image = item["observation.image"]  # (obs_horizon, C, H, W)

            episode_index = item["episode_index"]
            current_episode_end = self.lerobot_dataset.episode_data_index[
                "to"
            ][episode_index].item()

            # iteration at most until the end of the current episode
            max_horizon = current_episode_end - self.starting_idx
            if i + self.drop_last < max_horizon:
                # print(f"Enriching dataset: {i + self.starting_idx}/{max_horizon - 1 + self.starting_idx} - {current_episode_end}")
                future_goal_pos = self.dataset[max_horizon - 1][
                    "observation.state"
                ][-1]
                future_goal_image = self.dataset[max_horizon - 1][
                    "observation.image"
                ][-1]
                self.enriched_data.append(
                    {
                        "observation.state": agent_pos,
                        "action": action,
                        "observation.image": image,
                        "reached_goal.state": future_goal_pos,
                        "reached_goal.image": future_goal_image,
                    }
                )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.enriched_data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return self.enriched_data[idx]


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
    expert_dataset: Dataset,
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
    expert_demonstrations = expert_dataset
    demonstration_statistics = get_demonstration_statistics()
    num_episodes = min(num_episodes, expert_demonstrations.num_episodes)
    starting_episode = np.random.randint(
        0, expert_demonstrations.num_episodes - num_episodes + 1
    )
    print(f"Starting episode: {starting_episode}")
    from_idx = expert_demonstrations.episode_data_index["from"][
        starting_episode
    ].item()
    to_idx = expert_demonstrations.episode_data_index["to"][
        starting_episode + num_episodes - 1
    ].item()
    if cfg.expert_data.transitions == "subsequent":
        print("Building subsequent dataset...")
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
        print("Building terminal dataset...")
        # rollouts = [expert_demonstrations[idx] for idx in range(from_idx, to_idx)]
        # dataset = EnrichedTerminalRobotDataset(
        #     dataset=rollouts,
        #     lerobot_dataset=expert_demonstrations,
        #     starting_idx=from_idx,
        # )

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
            dataset = EnrichedTerminalRobotDataset2(
                dataset=rollout,
            )
            episodes_datasets.append(dataset)
        dataset = torch.utils.data.ConcatDataset(episodes_datasets)

        # rollouts = [
        #     expert_demonstrations[idx] for idx in range(from_idx, to_idx)
        # ]
        # dataset = EnrichedTerminalRobotDataset3(
        #     dataset=rollouts,
        #     lerobot_dataset=expert_demonstrations,
        #     starting_idx=from_idx,
        # )
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
