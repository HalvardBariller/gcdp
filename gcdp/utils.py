import numpy as np
import numba
import matplotlib.pyplot as plt
from copy import deepcopy
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import gym_pusht
import tqdm


def record_video(env, name, horizon=180, policy=None):
    """
    input
    horizon : length of the simulation
    policy : either a determinstic policy represented by an (H,S) array
    or a random policy which is uniform (None)
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
        except Exception as e:
            print(f"Error closing video recorder: {e}")

    # Ensure environment is properly closed
    try:
        env.close()
    except Exception as e:
        print(f"Error closing environment: {e}")

    print("Environment closed successfully.")

    print("Reward sum: {}".format(tot_reward))


class ScaleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_state, reward, term, trunc, info = self.env.step(action)
        if term:
            reward = 1.0
        else:
            reward = 0.0

        return (next_state, reward, term, trunc, info)



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