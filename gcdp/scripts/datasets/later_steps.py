"""Builds the dataset for predicting the future states along the trajectories."""

from copy import deepcopy
from omegaconf import DictConfig
from torch.utils.data import Dataset
from typing import List
import numpy as np
import torch
import tqdm

from gcdp.scripts.common.utils import get_demonstration_statistics
from gcdp.scripts.datasets.trajectory_expert import NormalizedExpertDataset


def build_successive_observations_dataset(
    cfg: DictConfig,
    expert_demonstrations: Dataset,
    timestep_shift: int = None,
    normalize: bool = True,
):
    """
    Build a dataset of observations and their future states from the given trajectories.

    Inputs:
        cfg : configuration file
        expert_dataset : the dataset of expert demonstrations
        timestep_shift : the index shift for the future state associated with each observation
        normalize : whether to normalize the dataset
    Outputs:
        dataset (Dataset): the dataset of goal-conditioned expert demonstrations
    """
    demonstration_statistics = get_demonstration_statistics()
    num_episodes = expert_demonstrations.num_episodes
    if timestep_shift is None:
        timestep_shift = cfg.model.pred_horizon * 2
    episodes_datasets = []
    for episode_index in tqdm.tqdm(num_episodes):
        from_idx = expert_demonstrations.episode_data_index["from"][
            episode_index
        ].item()
        to_idx = expert_demonstrations.episode_data_index["to"][
            episode_index
        ].item()
        rollout = [
            expert_demonstrations[idx] for idx in range(from_idx, to_idx)
        ]
        dataset = future_steps_dataset(
            rollout,
            shift=timestep_shift,
        )
        dataset = FutureObservationDataset(dataset=dataset)
        episodes_datasets.append(dataset)
    dataset = torch.utils.data.ConcatDataset(episodes_datasets)
    if normalize:
        dataset = NormalizedExpertDataset(
            expert_dataset=dataset,
            demonstration_statistics=demonstration_statistics,
        )
    return dataset


def future_steps_dataset(dataset, shift):
    """Create the dataset with future states for a single rollout (obtained from LeRobot original expert dataset).

    Parameters:
        dataset: a single episode from the expert dataset
        shift: the number of steps to look ahead for the future state
        -------------------------------------------
        | timestep      | t | t+1 | ... | t+shift |
        |---------------|---|-----|-----|---------|
        | observation   | Y | N   | ... | N       |
        | future        | N | N   | ... | Y       |
        -------------------------------------------
    """
    rollout = deepcopy(dataset)
    enriched_data = []

    for i in range(len(rollout)):
        item = rollout[i]
        image = item["observation.image"][-1]  # (C, H, W)
        # Pad the end of the rollout whose shift goes beyond horizon with the last state
        future_idx = min(i + shift, len(rollout) - 1)
        enriched_data.append(
            {
                "observation.image": image,  # (obs_horizon, C, H, W)
                "future.image": rollout[future_idx]["observation.image"][
                    -1
                ],  # (C, H, W)
            }
        )
    return enriched_data


class FutureObservationDataset(torch.utils.data.Dataset):
    """Build the dataset containing future observations for a single rollout."""

    def __init__(self, dataset):
        """Initialize the dataset.

        Inputs:
            dataset: a single enriched episode (obtained from future_steps_dataset)
        """
        self.enriched_data = deepcopy(dataset)

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
