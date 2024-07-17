"""This script contains the function to evaluate a policy in a Gym environment."""

import collections
import gymnasium as gym
import numpy as np
import pymunk
import shapely.geometry as sg
import tqdm

from copy import deepcopy
from gcdp.policy import diff_policy
from gymnasium.wrappers import RecordVideo
from pathlib import Path
from pymunk.vec2d import Vec2d


def eval_policy(
    env: gym.Env,
    num_episodes: int,
    max_steps: int,
    model,
    noise_scheduler,
    observations,
    device,
    network_params,
    normalization_stats,
    successes,
    save_video: bool = False,
    video_path: str = None,
    video_prefix: str = None,
    seed: int = 42,
):
    """
    Evaluate a policy over a specified number of episodes in a Gym environment.

    Parameters:
        env (gym.Env): The Gym environment to evaluate the policy in.
        num_episodes (int): Number of episodes to run the evaluation.
        max_steps (int): Maximum number of steps per episode.
        save_video (bool): If True, saves a video of the last episode.
        video_path (str): Directory path to save the video.
        video_prefix (str): Prefix for the video file name.
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
    # model = kwargs["model"]
    # noise_scheduler = kwargs["noise_scheduler"]
    # observations = kwargs["observations"]
    # device = kwargs["device"]
    # network_params = kwargs["network_params"]
    # normalization_stats = kwargs["normalization_stats"]
    # successes = kwargs["successes"]

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
        task_completed = False
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
                if done:
                    task_completed = True
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
        episode_results["success"].append(task_completed)
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


def compute_coverage(info):
    """
    Compute the coverage of the block over the goal area.

    This function is adapted from the Push-T environment:
    https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py

    Parameters:
        info (dict): Dictionary containing the block and goal pose information.
    Returns:
        float: The coverage of the block over the goal area.
    """

    def pymunk_to_shapely(body, shapes):
        """
        Convert a pymunk body with shapes to a shapely geometry.

        Parameters:
            body (pymunk.Body): The pymunk body.
            shapes (list): List of pymunk shapes attached to the body.
        Returns:
            shapely.geometry.base.BaseGeometry: The shapely geometry.
        """
        geoms = []
        for shape in shapes:
            if isinstance(shape, pymunk.shapes.Poly):
                verts = [body.local_to_world(v) for v in shape.get_vertices()]
                verts += [verts[0]]
                geoms.append(sg.Polygon(verts))
            else:
                raise RuntimeError(f"Unsupported shape type {type(shape)}")
        geom = sg.MultiPolygon(geoms)
        return geom

    # Extract the necessary information from the dictionary
    block_pose = info["block_pose"]
    goal_pose = info["goal_pose"]

    # Create a pymunk body for the block and the goal
    block_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    block_body.position = tuple(block_pose[:2])  # Convert numpy array to tuple
    block_body.angle = block_pose[2]

    goal_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    goal_body.position = tuple(goal_pose[:2])  # Convert numpy array to tuple
    goal_body.angle = goal_pose[2]

    # Create a shapely polygon for the block and the goal
    block_shape = pymunk.Poly.create_box(block_body, size=(50, 100))
    goal_shape = pymunk.Poly.create_box(goal_body, size=(50, 100))

    block_geom = pymunk_to_shapely(block_body, [block_shape])
    goal_geom = pymunk_to_shapely(goal_body, [goal_shape])

    # Compute the intersection area and the goal area
    intersection_area = goal_geom.intersection(block_geom).area
    goal_area = goal_geom.area

    # Compute the coverage
    coverage = intersection_area / goal_area

    return coverage


def eval_policy_on_interm_goals(
    env: gym.Env,
    num_episodes: int,
    max_steps: int,
    model,
    noise_scheduler,
    observations,
    device,
    network_params,
    normalization_stats,
    target,
    target_block_pose,
    save_video: bool = False,
    video_path: str = None,
    video_prefix: str = None,
    seed: int = 42,
):
    """
    Evaluate a policy over a specified number of episodes in a Gym environment on intermediate goals.

    Parameters:
        env (gym.Env): The Gym environment to evaluate the policy in.
        num_episodes (int): Number of episodes to run the evaluation.
        max_steps (int): Maximum number of steps per episode.
        save_video (bool): If True, saves a video of the last episode.
        video_path (str): Directory path to save the video.
        video_prefix (str): Prefix for the video file name.
        seed (int): Random seed for environment setup.
        model: Neural network model used for policy decision.
        noise_scheduler: Scheduler for noise process in policy execution.
        observations: Initial state observations for the policy.
        device: Computation device (CPU/GPU) for model operations.
        network_params (dict): Parameters specific to the neural network model.
        normalization_stats: Statistics for normalizing input data.
        target (dict): Image of the goal and agent pose.
        target_block_pose (np.ndarray): Block pose of the target goal.

    Returns:
        dict: A dictionary containing the success rate, average rewards, and details of the last goal.
    """
    # model = kwargs["model"]
    # noise_scheduler = kwargs["noise_scheduler"]
    # observations = kwargs["observations"]
    # device = kwargs["device"]
    # network_params = kwargs["network_params"]
    # normalization_stats = kwargs["normalization_stats"]

    actions_taken = network_params["action_horizon"]
    obs_horizon = network_params["obs_horizon"]

    episode_results = {
        "success": [],
        "rewards": [],
    }

    info_poses = {"goal_pose": target_block_pose}

    for episode in tqdm.tqdm(range(num_episodes)):
        if save_video and episode == num_episodes - 1:
            env = RecordVideo(
                env, video_path, disable_logger=True, name_prefix=video_prefix
            )
        seed += 1
        task_completed = False
        # Keep track of the planned actions
        action_queue = collections.deque(maxlen=actions_taken)
        # Randomly select a goal among the successful ones
        # goal = successes[np.random.randint(len_successes)]
        # Initialize the environment
        s, _ = env.reset(seed=seed)
        done = False
        tot_reward = 0
        observations = collections.deque([s] * obs_horizon, maxlen=obs_horizon)
        step = 0
        while not done:
            # Execute the planned actions
            if action_queue:
                s, _, _, _, inf = env.step(action_queue.popleft())
                step += 1
                info_poses["block_pose"] = inf["block_pose"]
                tot_reward += compute_coverage(info_poses)
                done = tot_reward > 0.9
                if done:
                    task_completed = True
            # Plan new actions
            else:
                action_chunk = diff_policy(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    observations=observations,
                    goal=target,
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
        episode_results["success"].append(task_completed)
        episode_results["rewards"].append(tot_reward)

    env.close()

    episode_results["success_rate"] = (
        sum(episode_results["success"]) / num_episodes
    )
    episode_results["average_reward"] = (
        sum(episode_results["rewards"]) / num_episodes
    )
    episode_results["last goal"] = target["pixels"]

    video_files = list(Path("video").rglob("*.mp4"))
    print("video files", video_files)
    if video_files:
        episode_results["rollout_video"] = video_files[0]

    return episode_results
