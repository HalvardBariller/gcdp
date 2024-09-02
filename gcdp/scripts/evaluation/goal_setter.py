"""This script contains functions for setting goals that will be used during evaluation."""

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from gcdp.scripts.datasets.trajectory_expert import load_expert_dataset


def first_state_expert_trajectory(expert_dataset: Dataset):
    """Load the first state of each expert trajectory.

    Returns a dictionary mapping the episode index to the first state of the episode.
    Parameters:
        expert_dataset (Dataset): Expert dataset.
    Returns:
        episodes (dict): Dictionary mapping the episode index to the first state of the episode (normalized).
    """
    episodes = {}
    # Map the episode index to the first state of the episode
    for episode_idx in range(expert_dataset.num_episodes):
        from_idx = expert_dataset.episode_data_index["from"][
            episode_idx
        ].item()
        episodes[episode_idx] = expert_dataset[from_idx]["observation.image"]
    return episodes


def closest_expert_trajectory(
    initial_state,
    expert_map,
    expert_dataset: Dataset,
    # num_goals: int = None,
):
    """Find the closest point among the expert trajectories to the initial state.

    Returns a list of states from the selected expert trajectory that will be used as the goal for evaluation.
    Parameters:
        initial_state (torch.Tensor): The initial state of the agent obtained from the environment.
        expert_map (dict): Dictionary mapping expert trajectories to their first states.
        cfg (DictConfig): Configuration file.
        expert_dataset (Dataset): Expert dataset.
        @TODO: Add num_goals parameter to return a fixed number of goals
        # num_goals (int): The number of goals to return. If None, return all goals.
    Returns:
        goals (list): Sequence of states from the expert trajectory.
    """
    initial_state = initial_state["pixels"]
    initial_state = np.moveaxis(initial_state, -1, 0)  # (C, H, W)
    initial_state = initial_state / 255.0
    initial_state = initial_state.flatten()
    min_distance = float("inf")
    closest_expert = None
    for expert_idx, expert_state in expert_map.items():
        expert_state = expert_state.cpu().numpy()
        expert_state = expert_state.flatten()
        distance = np.linalg.norm(initial_state - expert_state)
        if distance < min_distance:
            min_distance = distance
            closest_expert = expert_idx
    goals = []
    from_idx = expert_dataset.episode_data_index["from"][closest_expert].item()
    to_idx = expert_dataset.episode_data_index["to"][closest_expert].item()
    for idx in range(from_idx, to_idx):
        goal = expert_dataset[idx]["observation.image"] / 255.0
        goals.append(goal)
    return goals
