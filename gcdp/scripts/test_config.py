"""Train the model from a configuration file."""

import logging
import numpy as np
import pickle
import hydra
import os
import torch

from diffusers.optimization import get_scheduler
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler

from gcdp.model.modeling import (
    make_diffusion_model,
    make_optimizer_and_scheduler,
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


# def make_optimizer_and_scheduler(cfg, policy, num_training_examples):
#     """Create the optimizer and scheduler."""
#     if cfg.optim.name == "adam":
#         optimizer = torch.optim.Adam(
#             policy.parameters(),
#             lr=cfg.optim.lr,
#             eps=cfg.optim.eps,
#             betas=cfg.optim.betas,
#             weight_decay=cfg.optim.weight_decay,
#         )
#     elif cfg.optim.name == "adamw":
#         optimizer = torch.optim.AdamW(
#             policy.parameters(),
#             lr=cfg.optim.lr,
#             eps=cfg.optim.eps,
#             betas=cfg.optim.betas,
#             weight_decay=cfg.optim.weight_decay,
#         )
#     else:
#         raise ValueError(f"Optimizer {cfg.optim.name} not implemented.")
#     lr_scheduler = get_scheduler(
#         name=cfg.optim.scheduler.name,
#         optimizer=optimizer,
#         num_warmup_steps=cfg.optim.scheduler.num_warmup_steps,
#         num_training_steps=cfg.training.num_epochs * cfg.training.policy_refinement * num_training_examples,
#     )
#     return optimizer, lr_scheduler


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
    # TODO: optimizer, lr_scheduler = make_optimizer_and_scheduler(...)


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
