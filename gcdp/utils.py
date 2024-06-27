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


# def get_rollout(episode_length=50, policy=None, env=None):
#     """
#     input
#     episode_length : length of the simulation
#     policy : either a deterministic policy or a uniform random policy
#     env : gym environment
#     """
#     if env is None:
#         env = gym.make("gym_pusht/PushT-v0")
#     s, _ = env.reset()
#     desired_goal = env.observation_space.sample()
#     done = False
#     h = 0
#     states = []
#     actions = []
#     while not done:
#         if policy is not None:
#             # action = get_action(policy,
#             #                     state=s,
#             #                     goal=desired_goal,
#             #                     horizon ?)
#             raise NotImplementedError
#         else:
#             action = env.action_space.sample()
#         s, _, term, trunc, infos = env.step(action)
#         states.append(s)
#         actions.append(action)
#         h += 1
#         done = (term or trunc) or h >= episode_length
#     return {
#         "states": np.array(states),
#         "actions": np.array(actions),
#         "desired_goal": desired_goal,
#     }
