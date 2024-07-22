"""This module contains utility functions for the GCDP project."""

import os
from os import open
import pickle
import gymnasium as gym
import gym_pusht
import numpy as np
import random
import torch

from copy import deepcopy
from gymnasium.wrappers import RecordVideo


def record_video(env, name, horizon=180, policy=None):
    """
    Record a video of the environment for a given policy.

    Parameters:
        horizon : length of the simulation
        policy : either a determinstic policy represented by an (H,S) array or a random policy which is uniform (None)
    """
    env = deepcopy(env)
    video_folder = "./gym_videos/" + name
    env = RecordVideo(env, video_folder, disable_logger=True)
    s, _ = env.reset()
    done = False
    tot_reward = 0
    h = 0
    while not done:
        if policy is not None:
            raise NotImplementedError
        else:
            action = env.action_space.sample()
        s, r, term, trunc, infos = env.step(action)
        h += 1
        tot_reward += r
        done = (term or trunc) or h >= horizon

    # Close video recorder if it is still active
    if hasattr(env, "video_recorder") and env.video_recorder is not None:
        try:
            env.video_recorder.close()
        except AttributeError as e:
            print(f"Attribute error when closing video recorder: {e}")
        except IOError as e:
            print(f"IO error when closing video recorder: {e}")
        # except Exception as e:
        #     # Catch any other unexpected exceptions and re-raise
        #     print(f"Unexpected error closing video recorder: {e}")
        #     raise

    # Ensure environment is properly closed
    try:
        env.close()
    except AttributeError as e:
        print(f"Attribute error when closing environment: {e}")
    except IOError as e:
        print(f"IO error when closing environment: {e}")
    # except Exception as e:
    #     # Catch any other unexpected exceptions and re-raise
    #     print(f"Unexpected error closing environment: {e}")
    #     raise

    print("Environment closed successfully.")

    print("Reward sum: {}".format(tot_reward))


class ScaleRewardWrapper(gym.RewardWrapper):
    """This wrapper scales the reward to 1.0 for success and 0.0 otherwise."""

    def __init__(self, env):
        """Initialize the wrapper."""
        super().__init__(env)

    def step(self, action):
        next_state, reward, term, trunc, info = self.env.step(action)
        if term:
            reward = 1.0
        else:
            reward = 0.0

        return (next_state, reward, term, trunc, info)


def pusht_init_env(sparse_reward=True):
    """Initialize the environment for the PUSHT task."""
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    return env if not sparse_reward else ScaleRewardWrapper(env)


# normalize data
def get_data_stats(data):
    """Get the min and max values of the data."""
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    """Normalize the data to [-1, 1]."""
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
    """Unnormalize the data to the original range."""
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def set_global_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_demonstration_statistics():
    """Load the statistics of the demonstrations."""
    demonstration = np.load(
        "objects/demonstration_statistics.npz",
        allow_pickle=True,
    )
    demonstration_statistics = {
        key: demonstration[key].item() for key in demonstration
    }
    demonstration.close()
    return demonstration_statistics


# DOES NOT WORK WITH HYDRA
# def get_demonstration_successes(file_path):
#     with open(file_path, "rb") as f:
#         successes = pickle.load(f)
#     for item in successes:
#         item["pixels"] = item["pixels"].astype(np.float64)
#     return successes
