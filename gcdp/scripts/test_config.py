"""Train the model from a configuration file."""

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="torch.utils._pytree._register_pytree_node is deprecated",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="transformers.deepspeed module is deprecated and will be removed in a future version",
)

import logging
import numpy as np
import pickle
import hydra
import os
import torch
import torch.nn as nn
import tqdm

from diffusers.optimization import get_scheduler
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler


from gcdp.model.modeling import (
    make_diffusion_model,
    make_optimizer_and_scheduler,
)
from gcdp.scripts.episodes import (
    build_dataset,
    get_guided_rollout,
    get_random_rollout,
)
from gcdp.scripts.logger import init_logging, Logger, log_output_dir
from gcdp.scripts.utils import (
    get_demonstration_statistics,
    pusht_init_env,
    set_global_seed,
)


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
    """Compute the loss of the model."""
    obs_horizon = params["obs_horizon"]
    pred_horizon = params["pred_horizon"]
    device = cfg.device
    nimage = nbatch["image"][:, :obs_horizon].to(
        device
    )  # (B, obs_horizon, C, H, W) = (64, 2, 3, 96, 96)
    nagent_pos = nbatch["agent_pos"][:, :obs_horizon].to(
        device
    )  # (B, obs_horizon, 2) = (64, 2, 2)
    naction = nbatch["action"].to(device)  # (B, pred_horizon, 2) = (64, 16, 2)
    nreachedimage = nbatch["reached_goal_image"].to(
        device
    )  # (B, C, H, W) = (64, 3, 96, 96)
    nreachedagent_pos = nbatch["reached_goal_agent_pos"].to(
        device
    )  # (B, 2) = (64, 2)
    nreachedagent_pos = nreachedagent_pos.unsqueeze(
        1
    )  # (B, 1, 2) = (64, 1, 2)
    B = nagent_pos.shape[0]
    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
    # (B*obs_horizon, C, H, W) = (128, 3, 96, 96) --> (128, 512)
    image_features = image_features.reshape(
        *nimage.shape[:2], -1
    )  # (B, obs_horizon, D) = (64, 2, 512)

    reached_image_features = nets["vision_encoder"](
        nreachedimage
    )  # (B, C, H, W) = (64, 3, 96, 96) --> (64, 512)
    reached_image_features = reached_image_features.unsqueeze(
        1
    )  # (B, 1, D) = (64, 1, 512)

    # concatenate vision feature and low-dim obs
    obs_features = torch.cat(
        [image_features, nagent_pos], dim=-1
    )  # (B, obs_horizon, obs_dim) = (64, 2, 514)
    obs_cond = obs_features.flatten(
        start_dim=1
    )  # (B, obs_horizon * obs_dim) = (64, 1028)

    # concatenate vision goal feature and low-dim obs
    reached_obs_features = torch.cat(
        [reached_image_features, nreachedagent_pos], dim=-1
    )  # (B, 1, obs_dim) = (64, 1, 514)
    reached_obs_cond = reached_obs_features.flatten(
        start_dim=1
    )  # (B, obs_dim) = (64, 514)

    # concatenate obs and goal
    full_cond = torch.cat([obs_cond, reached_obs_cond], dim=-1)
    # (B, obs_horizon * obs_dim + goal_dim)  = (64, 1542)
    # Convert to float32
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

    loss = nn.functional.mse_loss(noise_pred, noise)
    return loss


def training_config(cfg: DictConfig, out_dir: str, job_name: str) -> None:
    """Training of the model."""
    init_logging()
    env = pusht_init_env(sparse_reward=cfg.env.sparse_reward)
    demonstration_statistics = get_demonstration_statistics()
    successes = get_demonstration_successes("objects/successes.pkl")
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
    set_global_seed(cfg.seed)
    log_output_dir(out_dir)
    grad_scaler = GradScaler(enabled=cfg.use_amp)
    logging.info("Building Diffusion Model.")
    nets, ema_nets, ema, noise_scheduler = make_diffusion_model(cfg)
    num_parameters_noise = sum(
        p.numel() for p in nets["noise_pred_net"].parameters()
    )
    num_parameters_vision_encoder = sum(
        p.numel() for p in nets["vision_encoder"].parameters()
    )
    num_trainable_parameters = sum(
        [
            sum(
                p.numel()
                for p in nets["noise_pred_net"].parameters()
                if p.requires_grad
            ),
            sum(
                p.numel()
                for p in nets["vision_encoder"].parameters()
                if p.requires_grad
            ),
        ]
    )
    logging.info(
        f"Number of parameters in Noise Model: {num_parameters_noise:,}"
    )
    logging.info(
        f"Number of parameters in Vision Encoder: {num_parameters_vision_encoder:,}"
    )
    logging.info(
        f"Number of trainable parameters: {num_trainable_parameters:,}"
    )
    logging.info("Training Diffusion Model.")
    params = build_params(cfg)
    for p in range(cfg.training.policy_refinement):
        logging.info(
            f"Policy Refinement {p + 1}/{cfg.training.policy_refinement}"
        )

        # Initial random rollout
        if p == 0:
            if cfg.data_generation.get_block_poses:
                trajectories = []
                for _ in range(cfg.data_generation.num_episodes):
                    trajectory, block_poses = get_random_rollout(
                        episode_length=cfg.data_generation.episode_length,
                        env=env,
                        get_block_poses=True,
                    )
                trajectories.append(trajectory)
            else:
                trajectories = [
                    get_random_rollout(
                        episode_length=cfg.data_generation.episode_length,
                        env=env,
                    )
                    for _ in range(cfg.data_generation.num_episodes)
                ]
            dataset = build_dataset(
                cfg, trajectories, demonstration_statistics
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            optimizer, lr_scheduler = make_optimizer_and_scheduler(
                cfg, nets, len(dataset)
            )
            logging.info(f"Number of training examples: {len(dataset)}")

        # Rollout with refined policy
        else:
            logging.info("Generating Rollouts with Refined Policy.")
            if cfg.data_generation.get_block_poses:
                trajectories = []
                for _ in range(cfg.data_generation.num_episodes):
                    trajectory, block_poses = get_guided_rollout(
                        episode_length=cfg.data_generation.episode_length,
                        env=env,
                        model=ema_nets,
                        device=cfg.device,
                        network_params=params,
                        normalization_stats=demonstration_statistics,
                        noise_scheduler=noise_scheduler,
                        get_block_poses=True,
                    )
                    trajectories.append(trajectory)
            else:
                trajectories = [
                    get_guided_rollout(
                        episode_length=cfg.data_generation.episode_length,
                        env=env,
                        model=ema_nets,
                        device=cfg.device,
                        network_params=params,
                        normalization_stats=demonstration_statistics,
                        noise_scheduler=noise_scheduler,
                    )
                    for _ in range(cfg.data_generation.num_episodes)
                ]
            dataset = build_dataset(
                cfg, trajectories, demonstration_statistics
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )

        step = 0
        with tqdm.tqdm(dataloader, desc="Batch", leave=False) as tepoch:
            for nbatch in tepoch:
                step += 1
                optimizer.zero_grad()
                loss = compute_loss(nbatch, params, nets, noise_scheduler, cfg)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                lr_scheduler.step()
                tepoch.set_postfix(loss=loss.item(), step=step)
                logger.log_training_step(step, loss.item())


@hydra.main(version_base="1.2", config_path="../config", config_name="config")
def train_cli(cfg: DictConfig) -> None:
    """Training from a configuration file."""
    training_config(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    train_cli()
