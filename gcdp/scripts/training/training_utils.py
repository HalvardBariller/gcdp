"""Define utility functions for training the model."""

import numpy as np
import pickle
import torch
import torch.nn as nn

from omegaconf import DictConfig


def get_demonstration_successes(file_path):
    """Load the successes of the demonstrations."""
    with open(file_path, "rb") as f:
        successes = pickle.load(f)
    for item in successes:
        item["pixels"] = item["pixels"].astype(np.float64)
    return successes


def build_params(cfg: DictConfig) -> dict:
    """Build the parameters of the network for later use."""
    params = {
        "obs_horizon": cfg.model.obs_horizon,
        "pred_horizon": cfg.model.pred_horizon,
        "action_dim": cfg.model.action_dim,
        "action_horizon": cfg.model.action_horizon,
        "num_diffusion_iters_train": cfg.diffusion.num_diffusion_iters_train,
        "num_diffusion_iters": cfg.diffusion.num_diffusion_iters_eval,
        "batch_size": cfg.training.batch_size,
        "policy_refinement": cfg.training.policy_refinement,
        "num_epochs": cfg.training.num_epochs,
        "episode_length": cfg.data_generation.episode_length,
        "num_episodes": cfg.data_generation.num_episodes,
        "seed": cfg.seed,
    }
    return params


def compute_loss(nbatch, params, nets, noise_scheduler, cfg):
    """Compute the loss of the model.

    Args:
        nbatch: Batch of data.
        params (dict): Parameters of the network.
        nets (nn.ModuleDict): Networks.
        noise_scheduler: Noise scheduler.
        cfg: Configuration file.
    """
    obs_horizon = params["obs_horizon"]
    pred_horizon = params["pred_horizon"]
    device = cfg.device
    nimage = nbatch["observation.image"].to(
        device
    )  # (B, obs_horizon, C, H, W)
    nagent_pos = nbatch["observation.state"].to(
        device
    )  # (B, obs_horizon, obs_dim)
    naction = nbatch["action"].to(device)  # (B, pred_horizon, action_dim)
    nreachedimage = nbatch["reached_goal.image"].to(device)  # (B, C, H, W)
    nreachedagent_pos = (
        nbatch["reached_goal.state"].to(device).unsqueeze(1)
    )  # (B, obs_dim) --> (B, 1, obs_dim)

    B = nagent_pos.shape[0]
    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
    # (B * obs_horizon, C, H, W) --> (B * obs_horizon, D)
    image_features = image_features.reshape(
        *nimage.shape[:2], -1
    )  # (B, obs_horizon, D)
    reached_image_features = nets["vision_encoder"](
        nreachedimage
    )  # (B, C, H, W) --> (B, D)
    reached_image_features = reached_image_features.unsqueeze(1)  # (B, 1, D)

    # concatenate vision feature and low-dim obs
    obs_features = torch.cat(
        [image_features, nagent_pos], dim=-1
    )  # (B, obs_horizon, D + obs_dim)
    obs_cond = obs_features.flatten(
        start_dim=1
    )  # (B, obs_horizon * (D + obs_dim))
    reached_goal_cond = reached_image_features.flatten(start_dim=1)  # (B, D)
    # concatenate obs and goal
    full_cond = (
        [obs_cond, reached_goal_cond]
        if cfg.model.goal_conditioned
        else [obs_cond]
    )
    full_cond = torch.cat(
        full_cond,
        dim=-1,
        # [obs_cond, reached_goal_cond], dim=-1
        # [obs_cond],
    )  # (B, obs_horizon * (D + obs_dim) + D)
    full_cond = full_cond.float()

    # sample noise to add to actions
    noise = torch.randn(naction.shape, device=device)
    # sample a diffusion iteration for each data point
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (B,),
        device=device,
    ).long()
    # add noise to clean images according to noise magnitude
    # at each diffusion iteration (forward diffusion process)
    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
    # predict the noise residual
    noise_pred = nets["noise_pred_net"](
        noisy_actions, timesteps, global_cond=full_cond
    )
    loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
    return loss.mean()
