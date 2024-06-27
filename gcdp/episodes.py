import gymnasium as gym
import gym_pusht
import numpy as np
import torch

# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from gcdp.utils import ScaleRewardWrapper


def get_random_rollout(episode_length=50, env=None):
    """
    Simulate an episode of the environment using the given policy.
    Inputs:
        episode_length : length of the simulation
        env : gym environment
    Outputs: dict containing the following keys:
        states : list of states (dicts) of the agent
        actions : list of actions taken by the agent
        reached_goals : list of states (dicts) of the agent after taking the actions
        desired_goal : the goal that the agent was trying to reach
    """
    if env is None:
        env = gym.make(
            "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
        )
        env = ScaleRewardWrapper(env)
    # desired_goal = env.observation_space.sample() # Random samp. had no meaning
    desired_goal, _ = env.reset()  # Reset is random

    s, _ = env.reset()
    states = []
    actions = []
    reached_goals = []
    for _ in range(episode_length):
        # if policy is not None:
        #     # action = get_action(policy,
        #     #                     state=s,
        #     #                     goal=desired_goal,
        #     #                     horizon ?)
        #     raise NotImplementedError
        action = env.action_space.sample()

        states.append(s)
        actions.append(action)
        s, _, _, _, _ = env.step(action)
        reached_goals.append(s)

    return {
        "states": states,
        "actions": actions,
        "reached_goals": reached_goals,
        "desired_goal": desired_goal,
    }


def split_trajectory(trajectory: dict):
    """
    Takes a trajectory of length H and splits it into its different components.
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
    reached_goals_pixels = np.array([t["pixels"] for t in trajectory["reached_goals"]])
    desired_goal_agent_pos = np.array(trajectory["desired_goal"]["agent_pos"])
    desired_goal_pixels = np.array(trajectory["desired_goal"]["pixels"])
    return {
        "states_agent_pos": states_agent_pos,
        "states_pixels": states_pixels,
        "actions": actions,
        "reached_goals_agent_pos": reached_goals_agent_pos,
        "reached_goals_pixels": reached_goals_pixels,
        "desired_goal_agent_pos": np.expand_dims(desired_goal_agent_pos, axis=0),
        "desired_goal_pixels": np.expand_dims(desired_goal_pixels, axis=0),
    }


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
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
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
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
    result = dict()
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
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # Convert entries to np arrays
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    for key in stats.keys():
        if not isinstance(stats[key], np.ndarray):
            stats[key] = np.array(stats[key])
    # normalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


class PushTDatasetFromTrajectories(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: list,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        get_original_goal: bool = False,
        dataset_statistics: dict = None,
    ):
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

        for t in trajectories:
            split = split_trajectory(t)
            train_image_data.append(split["states_pixels"])
            train_agent_pos_data.append(split["states_agent_pos"])
            train_action_data.append(split["actions"])
            train_reached_goals_agent_pos.append(split["reached_goals_agent_pos"])
            train_reached_goals_image.append(split["reached_goals_pixels"])
            train_desired_goal_agent_pos.append(split["desired_goal_agent_pos"])
            train_desired_goal_image.append(split["desired_goal_pixels"])

        # concatenate all trajectories (N transitions = sum over all T, T*H if
        # H is the length of each trajectory)
        train_image_data = np.concatenate(train_image_data, axis=0)
        # (N, 96, 96, 3)
        train_image_data = np.moveaxis(train_image_data, -1, 1)
        # (N, 3, 96, 96)
        train_reached_goals_image = np.concatenate(train_reached_goals_image, axis=0)
        # (N, 96, 96, 3)
        train_reached_goals_image = np.moveaxis(train_reached_goals_image, -1, 1)
        # (N, 3, 96, 96)
        train_goal_image_data = np.concatenate(
            train_desired_goal_image, axis=0
        )  # (T, 96, 96, 3)
        train_goal_image_data = np.moveaxis(
            train_goal_image_data, -1, 1
        )  # (T, 3, 96, 96)

        # Normalize images
        train_image_data = np.array(train_image_data).astype(np.float32) / 255.0
        train_reached_goals_image = (
            np.array(train_reached_goals_image).astype(np.float32) / 255.0
        )
        train_goal_image_data = (
            np.array(train_goal_image_data).astype(np.float32) / 255.0
        )

        if get_original_goal:
            self.train_data = {
                "agent_pos": np.concatenate(train_agent_pos_data, axis=0),  # (N, 2)
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
                "agent_pos": np.concatenate(train_agent_pos_data, axis=0),  # (N, 2)
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
                    self.train_data[key] = normalize_data(data, stats["action"])

    def __len__(self):
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
            desired_goal_agent_pos : np.ndarray of shape (1, 2)"""

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
