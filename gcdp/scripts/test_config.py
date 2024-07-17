"""Train the model from a configuration file."""

import numpy as np
import pickle
import hydra
import os

from omegaconf import DictConfig
import torch
from torch.cuda.amp import GradScaler

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


def make_optimizer_and_scheduler(cfg, policy):
    """Create the optimizer and scheduler."""
    if cfg.optim.name == "adam":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=cfg.optim.lr,
            eps=cfg.optim.eps,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
    elif cfg.optim.name == "adamw":
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=cfg.optim.lr,
            eps=cfg.optim.eps,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {cfg.optim.name} not implemented.")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma
    )
    return optimizer, scheduler


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
