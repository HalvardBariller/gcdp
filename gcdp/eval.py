"""This script contains the function to evaluate a policy in a Gym environment."""

import collections
import gym
import numpy as np
import tqdm
from gcdp.policy import diff_policy

from copy import deepcopy
from gymnasium.wrappers import RecordVideo
from pathlib import Path


def eval_policy(
    env: gym.Env,
    num_episodes: int,
    max_steps: int,
    save_video: bool = False,
    video_path: str = None,
    video_prefix: str = None,
    seed: int = 42,
    **kwargs,
):
    """
    Evaluate a policy over a specified number of episodes in a Gym environment.

    Parameters:
        env (gym.Env): The Gym environment to evaluate the policy in.
        num_episodes (int): Number of episodes to run the evaluation.
        max_steps (int): Maximum number of steps per episode.
        save_video (bool): If True, saves a video of the last episode.
        video_path (str): Directory path to save the video.
        video_fps (int): Frame rate of the saved video.
        seed (int): Random seed for environment setup.
        model: Neural network model used for policy decision.
        noise_scheduler: Scheduler for noise process in policy execution.
        observations: Initial state observations for the policy.
        device: Computation device (CPU/GPU) for model operations.
        network_params (dict): Parameters specific to the neural network model.
        normalization_stats: Statistics for normalizing input data.
        successes (list): List of successful outcomes to choose goals from.

    Returns:
        dict: A dictionary containing the success rate, average rewards, and details of the last goal.
    """
    model = kwargs["model"]
    noise_scheduler = kwargs["noise_scheduler"]
    observations = kwargs["observations"]
    device = kwargs["device"]
    network_params = kwargs["network_params"]
    normalization_stats = kwargs["normalization_stats"]

    successes = kwargs["successes"]
    len_successes = len(successes)

    actions_taken = network_params["action_horizon"]
    obs_horizon = network_params["obs_horizon"]

    episode_results = {
        "success": [],
        "rewards": [],
    }

    for episode in tqdm.tqdm(range(num_episodes)):
        if save_video and episode == num_episodes - 1:
            env = RecordVideo(
                env, video_path, disable_logger=True, name_prefix=video_prefix
            )
        seed += 1
        # Keep track of the planned actions
        action_queue = collections.deque(maxlen=actions_taken)
        # Randomly select a goal among the successful ones
        goal = successes[np.random.randint(len_successes)]
        # Initialize the environment
        s, _ = env.reset(seed=seed)
        done = False
        tot_reward = 0
        observations = collections.deque([s] * obs_horizon, maxlen=obs_horizon)
        step = 0
        while not done:
            # Execute the planned actions
            if action_queue:
                s, r, done, _, _ = env.step(action_queue.popleft())
                tot_reward += r
                step += 1
            # Plan new actions
            else:
                action_chunk = diff_policy(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    observations=observations,
                    goal=goal,
                    device=device,
                    network_params=network_params,
                    normalization_stats=normalization_stats,
                    actions_taken=actions_taken,
                )
                action_queue.extend(action_chunk)
            # Update the observations
            observations.append(s)
            if step > max_steps:
                done = True
        episode_results["success"].append(done)
        episode_results["rewards"].append(tot_reward)

    env.close()

    episode_results["success_rate"] = (
        sum(episode_results["success"]) / num_episodes
    )
    episode_results["average_reward"] = (
        sum(episode_results["rewards"]) / num_episodes
    )
    episode_results["last goal"] = goal["pixels"]

    video_files = list(Path("video").rglob("*.mp4"))
    print("video files", video_files)
    if video_files:
        episode_results["rollout_video"] = video_files[0]

    return episode_results
