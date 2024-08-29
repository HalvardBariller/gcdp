"""Modeling utilities for the GCDP model."""

import torch

from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch import nn
from omegaconf import DictConfig
from gcdp.model.diffusion import (
    ConditionalUnet1D,
    DiffusionRgbEncoder,
    get_resnet,
    replace_bn_with_gn,
)


def make_diffusion_model(cfg: DictConfig):
    """Create the diffusion model."""
    # vision_encoder = get_resnet(cfg.model.vision_encoder.name)
    # vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_encoder = DiffusionRgbEncoder(cfg)
    # @TODO: Verify this
    # vision_feature_dim = cfg.model.vision_encoder.feature_dim
    vision_feature_dim = (
        cfg.model.vision_encoder.spatial_softmax_num_keypoints * 2
    )
    agent_feature_dim = cfg.env.agent_pos_dim
    obs_dim = vision_feature_dim + agent_feature_dim
    obs_horizon = cfg.model.obs_horizon
    action_dim = cfg.model.action_dim
    # TODO: TEST WITHOUT INTEGRATING AGENT IN GOAL
    goal_dim = vision_feature_dim
    if cfg.model.goal_conditioned:
        global_cond_dim = obs_dim * obs_horizon + goal_dim
    else:
        global_cond_dim = obs_dim * obs_horizon
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
    )
    nets = nn.ModuleDict(
        {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
    )
    ema_nets = nets
    if cfg.diffusion.scheduler == "DDIM":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.diffusion.num_diffusion_iters_train,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            beta_schedule=cfg.diffusion.beta_schedule,
            clip_sample=cfg.diffusion.clip_sample,
            clip_sample_range=cfg.diffusion.clip_sample_range,
            prediction_type=cfg.diffusion.prediction_type,
        )
    elif cfg.diffusion.scheduler == "DDPM":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.diffusion.num_diffusion_iters_train,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            beta_schedule=cfg.diffusion.beta_schedule,
            clip_sample=cfg.diffusion.clip_sample,
            clip_sample_range=cfg.diffusion.clip_sample_range,
            prediction_type=cfg.diffusion.prediction_type,
        )
    device = cfg.device
    _ = nets.to(device)
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    _ = ema.to(device)
    return nets, ema_nets, ema, noise_scheduler


def make_optimizer_and_scheduler(cfg, model, num_batches):
    """Create the optimizer and scheduler."""
    if cfg.optim.name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optim.lr,
            eps=cfg.optim.eps,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
    elif cfg.optim.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            eps=cfg.optim.eps,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {cfg.optim.name} not implemented.")
    lr_scheduler = get_scheduler(
        name=cfg.optim.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.optim.scheduler.num_warmup_steps,
        num_training_steps=cfg.training.num_epochs
        * cfg.training.policy_refinement
        * num_batches,
    )
    return optimizer, lr_scheduler
