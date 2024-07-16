"""
This file contains functions that handle the generation of training data.

It includes functions to generate random or guided rollouts in the PushT
environment, to split the trajectories into different components and
to create a dataset of transitions from the trajectories.
"""

import collections
import gymnasium as gym
import numpy as np
import torch
import warnings

from diffusers import DDPMScheduler, DDIMScheduler

from gcdp.policy import diff_policy
from gcdp.utils import ScaleRewardWrapper, normalize_data


def get_random_rollout(episode_length=50, env=None, get_block_poses=False):
    """
    Simulate an episode of the environment using the given policy.

    Inputs:
        episode_length : length of the simulation
        env : gym environment
        get_block_poses : whether to return the block poses (used for evaluation on intermediary goals)
    Outputs:
        dict containing the following keys:
            states : list of states (dicts) of the agent
            actions : list of actions taken by the agent
            reached_goals : list of states (dicts) of the agent after taking the actions
            desired_goal : the goal that the agent was trying to reach
        list of block poses corresponding to coordinates of reached goals (only if get_block_poses is True)
    """
    if env is None:
        env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )
        env = ScaleRewardWrapper(env)
    # desired_goal = env.observation_space.sample() # Random samp. had no meaning
    desired_goal, _ = env.reset()  # Reset is random
    block_poses = []

    s, _ = env.reset()
    states = []
    actions = []
    reached_goals = []
    for _ in range(episode_length):
        action = env.action_space.sample()
        states.append(s)
        actions.append(action)
        s, _, _, _, infos = env.step(action)
        reached_goals.append(s)
        block_poses.append(infos["block_pose"])

    if get_block_poses:
        return {
            "states": states,
            "actions": actions,
            "reached_goals": reached_goals,
            "desired_goal": desired_goal,
        }, block_poses

    return {
        "states": states,
        "actions": actions,
        "reached_goals": reached_goals,
        "desired_goal": desired_goal,
    }


def get_guided_rollout(
    episode_length: int,
    env: gym.Env,
    model: torch.ModuleDict,
    device: torch.device,
    network_params: dict,
    normalization_stats: dict,
    noise_scheduler: DDPMScheduler | DDIMScheduler,
    get_block_poses: bool = False,
):
    """
    Simulate an episode of the environment using the given policy.

    Inputs:
        episode_length : length of the simulation
        env : gym environment
        model: the model used to predict the action (vision and noise models)
        noise_scheduler: the scheduler used to diffuse the noise
        device: the device to use
        network_params: the parameters of the network
        normalization_stats: the statistics used to normalize the data
        get_block_poses : whether to return the block poses (used for evaluation on intermediary goals)
    Outputs:
        dict containing the following
            states : list of states (dicts) of the agent
            actions : list of actions taken by the agent
            reached_goals : list of states (dicts) of the agent after taking the actions
            desired_goal : the goal that the agent was trying to reach
        list of block poses corresponding to coordinates of reached goals (only if get_block_poses is True)
    """
    if env is None:
        env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )
        env = ScaleRewardWrapper(env)
    desired_goal, _ = env.reset()  # Reset is random
    s, _ = env.reset()
    states = []
    actions = []
    reached_goals = []
    observations = collections.deque(
        [s] * network_params["obs_horizon"],
        maxlen=network_params["obs_horizon"],
    )
    block_poses = []
    for _ in range(episode_length):
        action = diff_policy(
            model=model,
            noise_scheduler=noise_scheduler,
            observations=observations,
            goal=desired_goal,
            device=device,
            network_params=network_params,
            normalization_stats=normalization_stats,
        )
        if action.shape != (2,):
            action = action[0]
        states.append(s)
        observations.append(s)
        actions.append(action)
        s, _, _, _, infos = env.step(action)
        reached_goals.append(s)
        block_poses.append(infos["block_pose"])

    if get_block_poses:
        return {
            "states": states,
            "actions": actions,
            "reached_goals": reached_goals,
            "desired_goal": desired_goal,
        }, block_poses

    return {
        "states": states,
        "actions": actions,
        "reached_goals": reached_goals,
        "desired_goal": desired_goal,
    }


def split_trajectory(trajectory: dict):
    """
    Take a trajectory of length H and splits it into its different components.

    Inputs:
        trajectory: dict containing the states, actions, reached goals and desired goal
                    (from get_rollout function for example)
    Outputs: dict containing the following keys:
        states_agent_pos: np.ndarray of shape (H, 2)
        states_pixels: np.ndarray of shape (H, 96, 96, 3)
        actions: np.ndarray of shape (H, 2)
        reached_goals_agent_pos: np.ndarray of shape (H, 2)
        reached_goals_pixels: np.ndarray of shape (H, 96, 96, 3)
        desired_goal_agent_pos: np.ndarray of shape (1, 2)
        desired_goal_pixels: np.ndarray of shape (1, 96, 96, 3)
    """
    states_agent_pos = np.array([s["agent_pos"] for s in trajectory["states"]])
    states_pixels = np.array([s["pixels"] for s in trajectory["states"]])
    actions = np.array(trajectory["actions"])
    reached_goals_agent_pos = np.array(
        [t["agent_pos"] for t in trajectory["reached_goals"]]
    )
    reached_goals_pixels = np.array(
        [t["pixels"] for t in trajectory["reached_goals"]]
    )
    desired_goal_agent_pos = np.array(trajectory["desired_goal"]["agent_pos"])
    desired_goal_pixels = np.array(trajectory["desired_goal"]["pixels"])
    return {
        "states_agent_pos": states_agent_pos,
        "states_pixels": states_pixels,
        "actions": actions,
        "reached_goals_agent_pos": reached_goals_agent_pos,
        "reached_goals_pixels": reached_goals_pixels,
        "desired_goal_agent_pos": np.expand_dims(
            desired_goal_agent_pos, axis=0
        ),
        "desired_goal_pixels": np.expand_dims(desired_goal_pixels, axis=0),
    }


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    # This function was adapted from the implementation of Chi et al. (2021)
    # https://diffusion-policy.cs.columbia.edu/
    """Create indices for sampling sequences from the trajectories."""
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = (
                min(idx + sequence_length, episode_length) + start_idx
            )
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [
                    buffer_start_idx,
                    buffer_end_idx,
                    sample_start_idx,
                    sample_end_idx,
                ]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
    episode_ends,
):
    # This function was adapted from the implementation of Chi et al. (2021)
    # https://diffusion-policy.cs.columbia.edu/
    """Sample a sequence from the training data."""
    result = {}
    for key, input_arr in train_data.items():
        if "reached_goal" in key:
            # Take the reached goal at the end of the sequence
            result[key] = input_arr[buffer_end_idx - 1]
            continue
        if "desired_goal" in key:
            # Count the number of values in the episode_ends array that are less than buffer_start_idx
            # This will give us the index of the episode that the buffer_start_idx belongs to
            episode_idx = np.sum(episode_ends < buffer_start_idx)
            result[key] = input_arr[episode_idx]
            continue
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype,
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


class PushTDatasetFromTrajectories(torch.utils.data.Dataset):
    """Dataset class for the PushT environment."""

    def __init__(
        self,
        trajectories: list,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        get_original_goal: bool,
        dataset_statistics: dict,
    ):
        """Initialize the dataset."""
        self.trajectories = trajectories  # list of T rollouts
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        train_image_data = []
        train_agent_pos_data = []
        train_action_data = []
        train_reached_goals_agent_pos = []
        train_reached_goals_image = []
        train_desired_goal_agent_pos = []
        train_desired_goal_image = []

        # if len(trajectories) > 1:
        #     warnings.warn(
        #         "The dataset is created from multiple trajectories."
        #         "Enriching the dataset with subsequent goals is not supported.",
        #         UserWarning,
        #         stacklevel=1,
        #     )

        for t in trajectories:
            split = split_trajectory(t)
            train_image_data.append(split["states_pixels"])
            train_agent_pos_data.append(split["states_agent_pos"])
            train_action_data.append(split["actions"])
            train_reached_goals_agent_pos.append(
                split["reached_goals_agent_pos"]
            )
            train_reached_goals_image.append(split["reached_goals_pixels"])
            train_desired_goal_agent_pos.append(
                split["desired_goal_agent_pos"]
            )
            train_desired_goal_image.append(split["desired_goal_pixels"])

        # concatenate all trajectories (N transitions = sum over all T, T*H if
        # H is the length of each trajectory)
        train_image_data = np.concatenate(train_image_data, axis=0)
        # (N, 96, 96, 3)
        train_image_data = np.moveaxis(train_image_data, -1, 1)
        # (N, 3, 96, 96)
        train_reached_goals_image = np.concatenate(
            train_reached_goals_image, axis=0
        )
        # (N, 96, 96, 3)
        train_reached_goals_image = np.moveaxis(
            train_reached_goals_image, -1, 1
        )
        # (N, 3, 96, 96)
        train_goal_image_data = np.concatenate(
            train_desired_goal_image, axis=0
        )  # (T, 96, 96, 3)
        train_goal_image_data = np.moveaxis(
            train_goal_image_data, -1, 1
        )  # (T, 3, 96, 96)

        # Normalize images
        train_image_data = (
            np.array(train_image_data).astype(np.float32) / 255.0
        )
        train_reached_goals_image = (
            np.array(train_reached_goals_image).astype(np.float32) / 255.0
        )
        train_goal_image_data = (
            np.array(train_goal_image_data).astype(np.float32) / 255.0
        )

        if get_original_goal:
            self.train_data = {
                "agent_pos": np.concatenate(
                    train_agent_pos_data, axis=0
                ),  # (N, 2)
                "action": np.concatenate(train_action_data, axis=0),  # (N, 2)
                "image": train_image_data,  # (N, 3, 96, 96)
                "reached_goal_agent_pos": np.concatenate(
                    train_reached_goals_agent_pos, axis=0
                ),  # (N, 2)
                "reached_goal_image": train_reached_goals_image,  # (N, 3, 96, 96)
                "desired_goal_agent_pos": np.concatenate(
                    train_desired_goal_agent_pos, axis=0
                ),  # (T, 2)
                "desired_goal_image": train_goal_image_data,  # (T, 3, 96, 96)
            }
        else:
            self.train_data = {
                "agent_pos": np.concatenate(
                    train_agent_pos_data, axis=0
                ),  # (N, 2)
                "action": np.concatenate(train_action_data, axis=0),  # (N, 2)
                "image": train_image_data,  # (N, 3, 96, 96)
                "reached_goal_agent_pos": np.concatenate(
                    train_reached_goals_agent_pos, axis=0
                ),  # (N, 2)
                "reached_goal_image": train_reached_goals_image,  # (N, 3, 96, 96)
            }

        # compute episode ends as the length of each trajectory
        self.episode_ends = np.cumsum([len(t["states"]) for t in trajectories])

        # compute start and end of each state-action sequence
        # also handles padding
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            # pad_before=0,
            pad_after=action_horizon - 1,
            # pad_after=0,
        )

        # compute the number of transitions for each episode
        self.episode_lengths = [
            np.sum(self.indices[:, 0] < end) for end in self.episode_ends
        ]

        # compute statistics and normalized data to [-1,1]
        if dataset_statistics is not None:
            stats = dataset_statistics
            for key, data in self.train_data.items():
                if "image" in key:
                    continue
                if "agent_pos" in key:
                    self.train_data[key] = normalize_data(
                        data, stats["observation.state"]
                    )
                if "action" in key:
                    self.train_data[key] = normalize_data(
                        data, stats["action"]
                    )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Inputs:
            idx : index of the sample
        Outputs: dict containing the following keys:
            image : np.ndarray of shape (obs_horizon, C, H, W)
            agent_pos : np.ndarray of shape (obs_horizon, 2)
            action : np.ndarray of shape (pred_horizon, 2)
            reached_goal_image : np.ndarray of shape (1, 3, 96, 96)
            reached_goal_agent_pos : np.ndarray of shape (1, 2)
            desired_goal_image : np.ndarray of shape (1, 3, 96, 96)
            desired_goal_agent_pos : np.ndarray of shape (1, 2)
        """
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )
        # get data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
            episode_ends=self.episode_ends,
        )
        # discard unused observations
        nsample["image"] = nsample["image"][: self.obs_horizon, :]
        nsample["agent_pos"] = nsample["agent_pos"][: self.obs_horizon, :]

        return nsample


class EnrichedDataset(torch.utils.data.Dataset):
    """Enriched dataset that includes all subsequent goals of the original dataset."""

    def __init__(self, original_dataset):
        """Initialize the dataset.

        Inputs:
            original_dataset: the original dataset to enrich (created w/ PushTDatasetFromTrajectories)
        """
        self.original_dataset = original_dataset
        self.enriched_data = []
        self.episode_lengths = original_dataset.episode_lengths
        self._enrich_dataset()

    def _enrich_dataset(self):
        n = len(self.original_dataset)
        for i in range(n):
            item = self.original_dataset[i]
            agent_pos = item["agent_pos"]
            action = item["action"]
            image = item["image"]
            reached_goal_agent_pos = item["reached_goal_agent_pos"]
            reached_goal_image = item["reached_goal_image"]

            # Original entry
            self.enriched_data.append(
                {
                    "agent_pos": agent_pos,
                    "action": action,
                    "image": image,
                    "reached_goal_agent_pos": reached_goal_agent_pos,
                    "reached_goal_image": reached_goal_image,
                }
            )

            # Add duplicate entries for subsequent goals
            current_episode_end = next(
                episode_end
                for episode_end in self.episode_lengths
                if i < episode_end
            )
            for j in range(i + 1, current_episode_end):
                # print(f"Enriching dataset: {i}/{n} - {j}/{n}")
                future_item = self.original_dataset[j]
                future_goal_pos = future_item["reached_goal_agent_pos"]
                future_goal_image = future_item["reached_goal_image"]
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
