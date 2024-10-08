"""This module contains the policy function that predicts the actions to take to reach the goal."""

import collections
import numpy as np
from omegaconf import DictConfig
import torch

from gcdp.scripts.common.utils import normalize_data, unnormalize_data
from diffusers import DDPMScheduler, DDIMScheduler


def diff_policy(
    model: torch.ModuleDict,
    noise_scheduler: DDPMScheduler | DDIMScheduler,
    observations: collections.deque,
    goal: dict,
    device: torch.device,
    network_params: dict,
    normalization_stats: dict,
    actions_taken: int,
    goal_conditioned: bool = True,
    goal_preprocessed: bool = False,
):
    """
    Predict a sequence of actions to take to reach the goal considering past observations.

    Inputs:
    - model: the model used to predict the action (vision and noise models)
    - noise_scheduler: the scheduler used to diffuse the noise
    - observations: the past observations
    - goal: the goal to reach (image already normalized)
    - device: the device to use
    - network_params: the parameters of the network
    - normalization_stats: the statistics used to normalize the data
    - actions_taken: the number of actions to execute
    - goal_conditioned: whether the policy is conditioned on the goal
    - goal_preprocessed: whether the goal is preprocessed or not (preprocessed when curriculum learning is used)
    Outputs:
    - actions: sequence of actions to execute
    """
    # Unpack network parameters
    obs_horizon = network_params["obs_horizon"]
    pred_horizon = network_params["pred_horizon"]
    action_dim = network_params["action_dim"]
    num_diffusion_iters = network_params["num_diffusion_iters"]

    images = np.stack(
        [x["pixels"] for x in observations]
    )  # (obs_horizon, H, W, C)
    images = np.moveaxis(images, -1, 1)  # (obs_horizon, C, H, W)
    agent_poses = np.stack(
        [x["agent_pos"] for x in observations]
    )  # (obs_horizon, action_dim)
    # Normalization
    images = images / 255.0
    agent_poses = normalize_data(
        agent_poses,
        stats=normalization_stats["observation.state"],
    )

    if not goal_preprocessed:
        goal_image = goal["pixels"]  # (H, W, C)
        goal_image = np.moveaxis(goal_image, -1, 0)  # (C, H, W)
        goal_image = goal_image / 255.0
        # goal_agent = goal["agent_pos"]  # (2,)
        # goal_agent = normalize_data(
        #     goal_agent,
        #     stats=normalization_stats["observation.state"],
        # )
    else:
        goal_image = goal.cpu().numpy()  # (C, H, W)

    # device transfer
    images = torch.from_numpy(images).to(device, dtype=torch.float32)
    agent_poses = torch.from_numpy(agent_poses).to(device, dtype=torch.float32)
    goal_image = torch.from_numpy(goal_image).to(device, dtype=torch.float32)
    goal_image = goal_image.unsqueeze(0)  # (1, C, H, W)
    # goal_agent = torch.from_numpy(goal_agent).to(device, dtype=torch.float32)
    # goal_agent = goal_agent.unsqueeze(0)  # (1, 2)

    # infer action
    with torch.no_grad():
        # get image features
        image_features = model["vision_encoder"](images)  # (obs_horizon, D)
        goal_image_features = model["vision_encoder"](goal_image)  # (1, D)
        # concat with low-dim observations
        obs_features = torch.cat(
            [image_features, agent_poses], dim=-1
        )  # (obs_horizon, D + obs_dim)
        goal_features = goal_image_features  # (1, D)

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(
            start_dim=1
        )  # (1, obs_horizon * (D + obs_dim))
        obs_cond = obs_cond.float()
        full_cond = (
            [obs_cond, goal_features] if goal_conditioned else [obs_cond]
        )
        full_cond = torch.cat(
            full_cond,
            dim=-1,
            # [obs_cond, goal_features], dim=-1
            # [obs_cond]
        )  # (1, obs_horizon * (D + obs_dim) + D)
        full_cond = full_cond.float()

        # initialize action from Gaussian noise
        noisy_action = torch.randn(
            (pred_horizon, action_dim),
            device=device,
        )  # (pred_horizon, action_dim)
        naction = noisy_action.unsqueeze(0)  # (1, pred_horizon, action_dim)
        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)
        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = model["noise_pred_net"](
                sample=naction, timestep=k, global_cond=full_cond
            )
            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
    # unnormalize action
    naction = naction.detach().to("cpu").numpy()
    # (B, pred_horizon, action_dim)
    naction = naction[0]
    action_pred = unnormalize_data(
        naction,
        stats=normalization_stats["action"],
    )
    start = obs_horizon - 1
    end = start + actions_taken
    actions = action_pred[start:end, :]  # (actions_taken, action_dim)
    return actions
