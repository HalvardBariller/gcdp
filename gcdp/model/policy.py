"""This module contains the policy function that predicts the actions to take to reach the goal."""

import collections
import numpy as np
import torch

from gcdp.scripts.utils import normalize_data, unnormalize_data
from diffusers import DDPMScheduler, DDIMScheduler


def diff_policy(
    model: torch.ModuleDict,
    noise_scheduler: DDPMScheduler | DDIMScheduler,
    observations: collections.deque,
    goal: dict,
    device: torch.device,
    network_params: dict,
    normalization_stats: dict,
    actions_taken: int = 1,
):
    """
    Predict a sequence of actions to take to reach the goal considering past observations.

    Inputs:
    - model: the model used to predict the action (vision and noise models)
    - noise_scheduler: the scheduler used to diffuse the noise
    - observations: the past observations
    - goal: the goal to reach
    - device: the device to use
    - network_params: the parameters of the network
    - normalization_stats: the statistics used to normalize the data
    - actions_taken: the number of actions to execute
    Outputs:
    - actions: sequence of actions to execute
    """
    # Unpack network parameters
    obs_horizon = network_params["obs_horizon"]
    pred_horizon = network_params["pred_horizon"]
    action_dim = network_params["action_dim"]
    num_diffusion_iters = network_params["num_diffusion_iters"]

    images = np.stack([x["pixels"] for x in observations])  # (2, 96, 96, 3)
    images = np.moveaxis(images, -1, 1)  # (2, 3, 96, 96)
    agent_poses = np.stack([x["agent_pos"] for x in observations])  # (2, 2)
    goal_image = goal["pixels"]  # (96, 96, 3)
    goal_image = np.moveaxis(goal_image, -1, 0)  # (3, 96, 96)
    goal_agent = goal["agent_pos"]  # (2,)

    # Normalization
    images = images / 255.0
    agent_poses = normalize_data(
        agent_poses,
        stats=normalization_stats["observation.state"],
    )
    goal_image = goal_image / 255.0
    goal_agent = normalize_data(
        goal_agent,
        stats=normalization_stats["observation.state"],
    )

    # device transfer
    images = torch.from_numpy(images).to(device, dtype=torch.float32)
    # (2, 3, 96, 96)
    agent_poses = torch.from_numpy(agent_poses).to(device, dtype=torch.float32)
    # (2, 2)
    goal_image = torch.from_numpy(goal_image).to(device, dtype=torch.float32)
    goal_image = goal_image.unsqueeze(0)  # (1, 3, 96, 96)
    goal_agent = torch.from_numpy(goal_agent).to(device, dtype=torch.float32)
    goal_agent = goal_agent.unsqueeze(0)  # (1, 2)

    # infer action
    with torch.no_grad():
        # get image features
        image_features = model["vision_encoder"](images)  # (2, 512)
        goal_image_features = model["vision_encoder"](goal_image)  # (1, 512)

        # concat with low-dim observations
        obs_features = torch.cat(
            [image_features, agent_poses], dim=-1
        )  # (2, 514)
        goal_features = torch.cat(
            [goal_image_features, goal_agent], dim=-1
        )  # (1, 514)

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)  # (1, 1028)
        full_cond = torch.cat([obs_cond, goal_features], dim=-1)  # (1, 1542)
        full_cond = full_cond.float()

        # initialize action from Gaussian noise
        noisy_action = torch.randn(
            (pred_horizon, action_dim),
            device=device,
        )  # (16, 2)
        naction = noisy_action.unsqueeze(0)  # (1, 16, 2)

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
