"""Transition datasets from expert demonstrations with different strategies for goal enrichment."""

import numpy as np
import torch

from copy import deepcopy


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
                f"action_horizon ({action_horizon}) - obs_horizon ({obs_horizon})"
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


class EnrichedEvenlySpacedRobotDataset(torch.utils.data.Dataset):
    """Enriched dataset of a single rollout using evenly spaced states along the trajectory as goals."""

    def __init__(self, dataset, num_goals, drop_n_last_frames=7):
        """Initialize the dataset.

        Inputs:
            dataset: a single episode to enrich (obtained from LeRobot original expert dataset)
            num_goals: the number of evenly spaced goals to add to the dataset
            drop_n_last_frames: the number of last frames to drop from the episode
        """
        self.rollout = deepcopy(dataset)
        self.num_goals = num_goals
        self.drop_last = drop_n_last_frames
        self.enriched_data = self._enrich_dataset()

    def _enrich_dataset(self):
        """Build the transition dataset using linearly spaced states between the end of the prediction horizon and the end of the episode as goals."""
        assert self.num_goals > 0, "Number of goals must be greater than 0"
        self.enriched_data = []
        for i in range(len(self.rollout) - self.drop_last):
            item = self.rollout[i]
            agent_pos = item["observation.state"]  # (obs_horizon, obs_dim)
            action = item["action"]  # (pred_horizon, action_dim)
            image = item["observation.image"]  # (obs_horizon, C, H, W)
            action_horizon = action.shape[0]
            obs_horizon = agent_pos.shape[0]

            if i + action_horizon - obs_horizon + 1 < len(self.rollout):
                # evenly spaced goals between the end of the prediction horizon and the end of the episode
                goals_idx = (
                    np.linspace(
                        i + action_horizon - obs_horizon + 1,
                        len(self.rollout) - 1,
                        self.num_goals,
                    )
                    .round()
                    .astype(int)
                )
            else:
                # last state as goal due to overlap with prediction horizon
                goals_idx = [len(self.rollout) - 1] * self.num_goals
            for j in goals_idx:
                # print(f"Enriching dataset: {i}/{len(self.rollout) - 1} - {j}/{len(self.rollout) - 1}")
                self.enriched_data.append(
                    {
                        "observation.state": agent_pos,
                        "action": action,
                        "observation.image": image,
                        "reached_goal.state": self.rollout[j][
                            "observation.state"
                        ][-1],
                        "reached_goal.image": self.rollout[j][
                            "observation.image"
                        ][-1],
                    }
                )
        return self.enriched_data

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.enriched_data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample_dict = self.enriched_data[idx]
        tensor_dict = {
            key: (
                value.clone().detach()
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, dtype=torch.float32)
            )
            for key, value in sample_dict.items()
        }
        return tensor_dict


def custom_collate_fn(batch):
    """Define a custom collate function for enriched datasets."""
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = torch.stack([item[key] for item in batch])
    return collated_batch


# class EnrichedTerminalRobotDataset3(torch.utils.data.Dataset):
#     """Enriched dataset that includes subsequent goals from the original expert dataset."""

#     def __init__(
#         self,
#         dataset,
#         lerobot_dataset,
#         starting_idx,
#         drop_n_last_frames=7,
#     ):
#         """Initialize the dataset.

#         Inputs:
#             dataset: the dataset to enrich (obtained from LeRobot, usually sliced from the original expert dataset)
#             goal_horizon: the number of steps to look ahead for the goal (must be at least action_horizon - obs_horizon + 1)
#             lerobot_dataset: the original expert dataset
#             starting_idx: the index in the original expert dataset of the first element in dataset
#             num_padding: the number of end-of-sequence padding transitions to add to the enriched dataset
#         """
#         self.dataset = dataset
#         self.enriched_data = []
#         self.lerobot_dataset = lerobot_dataset
#         self.starting_idx = starting_idx
#         self.drop_last = drop_n_last_frames
#         self._enrich_dataset()

#     def _enrich_dataset(self):
#         n = len(self.dataset)
#         cnt_padd = 0
#         for i in range(n):
#             item = self.dataset[i]
#             agent_pos = item["observation.state"]  # (obs_horizon, obs_dim)
#             action = item["action"]  # (pred_horizon, action_dim)
#             image = item["observation.image"]  # (obs_horizon, C, H, W)

#             episode_index = item["episode_index"]
#             current_episode_end = self.lerobot_dataset.episode_data_index[
#                 "to"
#             ][episode_index].item()

#             # iteration at most until the end of the current episode
#             max_horizon = current_episode_end - self.starting_idx
#             if i + self.drop_last < max_horizon:
#                 # print(f"Enriching dataset: {i + self.starting_idx}/{max_horizon - 1 + self.starting_idx} - {current_episode_end}")
#                 future_goal_pos = self.dataset[max_horizon - 1][
#                     "observation.state"
#                 ][-1]
#                 future_goal_image = self.dataset[max_horizon - 1][
#                     "observation.image"
#                 ][-1]
#                 self.enriched_data.append(
#                     {
#                         "observation.state": agent_pos,
#                         "action": action,
#                         "observation.image": image,
#                         "reached_goal.state": future_goal_pos,
#                         "reached_goal.image": future_goal_image,
#                     }
#                 )

#     def __len__(self):
#         """Return the number of samples in the dataset."""
#         return len(self.enriched_data)

#     def __getitem__(self, idx):
#         """Get a sample from the dataset."""
#         return self.enriched_data[idx]


# class EnrichedEvenlySpacedRobotDataset2(torch.utils.data.Dataset):
#     """Enriched dataset of a single rollout using evenly spaced states along the trajectory as goals."""

#     def __init__(self, dataset, num_goals, drop_n_last_frames=7):
#         """Initialize the dataset.

#         Inputs:
#             dataset: a single episode to enrich (obtained from LeRobot original expert dataset)
#             num_goals: the number of evenly spaced goals to add to the dataset
#             drop_n_last_frames: the number of last frames to drop from the episode
#         """
#         self.rollout = dataset
#         self.num_goals = num_goals
#         self.drop_last = drop_n_last_frames
#         self.enriched_data = []
#         self._enrich_dataset()
#         self.enriched_array = np.array(self.enriched_data)

#     def _enrich_dataset(self):
#         """Build the transition dataset using linearly spaced states between the end of the prediction horizon and the end of the episode as goals."""
#         assert self.num_goals > 0, "Number of goals must be greater than 0"
#         for i in range(len(self.rollout) - self.drop_last):
#             item = self.rollout[i]
#             agent_pos = item["observation.state"]  # (obs_horizon, obs_dim)
#             action = item["action"]  # (pred_horizon, action_dim)
#             image = item["observation.image"]  # (obs_horizon, C, H, W)
#             action_horizon = action.shape[0]
#             obs_horizon = agent_pos.shape[0]

#             if i + action_horizon - obs_horizon + 1 < len(self.rollout):
#                 # evenly spaced goals between the end of the prediction horizon and the end of the episode
#                 goals_idx = np.linspace(
#                     i + action_horizon - obs_horizon + 1, len(self.rollout) - 1, self.num_goals
#                 ).round().astype(int)
#             else:
#                 # last state as goal due to overlap with prediction horizon
#                 goals_idx = [len(self.rollout) - 1] * self.num_goals
#             for j in goals_idx:
#                 # print(f"Enriching dataset: {i}/{len(self.rollout) - 1} - {j}/{len(self.rollout) - 1}")
#                 self.enriched_data.append(
#                     {
#                         "observation.state": agent_pos,
#                         "action": action,
#                         "observation.image": image,
#                         "reached_goal.state": self.rollout[j]["observation.state"][-1],
#                         "reached_goal.image": self.rollout[j]["observation.image"][-1],
#                     }
#                 )

#     def __len__(self):
#         """Return the number of samples in the dataset."""
#         return len(self.enriched_data)

#     def __getitem__(self, idx):
#         """Get a sample from the dataset."""
#         try:
#             # return self.enriched_data[idx]
#             return self.enriched_array[idx]
#         except Exception:
#             print(f"Error at index {idx}")
#             print(f"Dataset length: {len(self.enriched_data)}")
#             print(f"Rollout length: {len(self.rollout)}")
#             raise
