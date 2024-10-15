"""Load an expert model and pursue the training using self-play and relabelling."""

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
import random
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import time
import tqdm

from diffusers.optimization import get_scheduler
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


from gcdp.model.modeling import (
    make_diffusion_model,
    make_optimizer_and_scheduler,
)
from gcdp.scripts.common.logger import (
    init_logging,
    Logger,
    log_eval_info,
    log_output_dir,
    log_train_info,
)
from gcdp.scripts.common.utils import (
    get_demonstration_statistics,
    pusht_init_env,
    set_global_seed,
)
from gcdp.scripts.common.visualisations import (
    aggregated_goal_map_visualisation,
    goal_map_visualisation,
)
from gcdp.scripts.datasets.episodes import (
    build_dataset,
    get_guided_rollout,
    get_random_rollout,
)
from gcdp.scripts.datasets.expert_datasets import custom_collate_fn
from gcdp.scripts.evaluation.eval import (
    eval_policy,
    eval_policy_on_interm_goals,
)
from gcdp.scripts.datasets.trajectory_expert import (
    batch_normalize_expert_input,
    build_expert_dataset,
    load_expert_dataset,
)
from gcdp.scripts.training.training_utils import (
    compute_loss,
    build_params,
    get_demonstration_successes,
)


def training_config(cfg: DictConfig, out_dir: str, job_name: str) -> None:
    """Training of the model."""
    init_logging()
    env = pusht_init_env(sparse_reward=cfg.env.sparse_reward)
    demonstration_statistics = get_demonstration_statistics()
    successes = get_demonstration_successes("objects/successes.pkl")
    designed_success = get_demonstration_successes(
        "objects/designed_success.pkl"
    )
    expert_dataset = load_expert_dataset(cfg)
    evaluation_expert_dataset = load_expert_dataset(cfg, timestamps=False)
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
    step = 0
    evaluation_results = {
        "max_reward": [],
        "success": [],
    }
    for p in range(cfg.training.policy_refinement):
        logging.info(
            f"Policy Refinement {p + 1}/{cfg.training.policy_refinement}"
        )
        # Initial random rollout
        if p == 0:
            dataset = build_expert_dataset(
                cfg,
                expert_dataset,
                cfg.expert_data.num_episodes,
            )
            logging.info(f"Number of training examples: {len(dataset)}")
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                collate_fn=custom_collate_fn,
                persistent_workers=(
                    True if cfg.training.num_workers > 0 else False
                ),
            )
            optimizer, lr_scheduler = make_optimizer_and_scheduler(
                cfg, nets, num_batches=len(dataloader)
            )
        # Rollout with refined policy
        else:
            if cfg.expert_data.num_episodes != expert_dataset.num_episodes:
                dataset = build_expert_dataset(
                    cfg,
                    expert_dataset,
                    cfg.expert_data.num_episodes,
                )
                logging.info(
                    f"Number of training examples: {len(dataset)}"
                )  # 24,208?
                dataloader = DataLoader(
                    dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=True,
                    num_workers=cfg.training.num_workers,
                    pin_memory=True,
                    collate_fn=custom_collate_fn,
                    persistent_workers=(
                        True if cfg.training.num_workers > 0 else False
                    ),
                )
        # Training
        with tqdm.tqdm(
            range(cfg.training.num_epochs), desc="Epoch", leave=False
        ) as tglobal:
            nets.train()
            for nepoch in tglobal:
                logging.info(f"Epoch {nepoch + 1}/{cfg.training.num_epochs}")
                epoch_losses = []
                with tqdm.tqdm(
                    dataloader, desc="Batch", leave=False
                ) as tepoch:
                    for nbatch in tepoch:
                        step += 1
                        optimizer.zero_grad()
                        start_time = time.perf_counter()
                        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                            loss = compute_loss(
                                nbatch, params, nets, noise_scheduler, cfg
                            )
                        grad_scaler.scale(loss).backward()
                        grad_scaler.unscale_(optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            nets.parameters(), cfg.optim.grad_clip_norm
                        )
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        ema.step(nets.parameters())
                        info = {
                            "loss": loss.item(),
                            "grad_norm": float(grad_norm),
                            "lr": optimizer.param_groups[0]["lr"],
                            "step_time": time.perf_counter() - start_time,
                            "policy_refinement": p + 1,
                            "epoch": nepoch + 1,
                            "step": step,
                        }
                        if step % cfg.training.log_interval == 0:
                            log_train_info(logger, info, step, cfg)
                        epoch_losses.append(loss.item())
            log_epoch = [
                f"Epoch: {nepoch + 1}",
                f"Epoch Loss: {np.mean(epoch_losses):.4f}",
            ]
            logging.info(" | ".join(log_epoch))

        # Weights of the EMA model for inference
        ema_nets = nets
        ema.copy_to(ema_nets.parameters())

        # Cleaning before next policy refinement
        if cfg.expert_data.num_episodes != expert_dataset.num_episodes:
            torch.cuda.empty_cache()
            del dataloader

        if cfg.save_model:
            logging.info("Saving Model.")
            torch.save(
                ema_nets.state_dict(),
                os.path.join(out_dir, "diffusion_model.pth"),
            )
            logging.info("Model Saved.")

        if cfg.evaluation.eval:
            logging.info("Evaluating Model.")
            env = pusht_init_env(sparse_reward=cfg.env.sparse_reward)
            video_path = "video/pusht/" + job_name
            video_prefix = "pusht_policy_refinement_" + str(p)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.use_amp):
                eval_results = eval_policy(
                    env=env,
                    num_episodes=cfg.evaluation.num_episodes,
                    max_steps=cfg.evaluation.max_steps,
                    save_video=cfg.evaluation.save_video,
                    video_path=video_path,
                    video_prefix=video_prefix,
                    seed=cfg.seed,
                    model=ema_nets,
                    # model=nets,
                    noise_scheduler=noise_scheduler,
                    observations=cfg.model.obs_horizon,
                    device=cfg.device,
                    network_params=params,
                    normalization_stats=demonstration_statistics,
                    successes=designed_success,  # @TODO or successes=successes,
                    progressive_goals=cfg.evaluation.progressive_goals,
                    expert_dataset=evaluation_expert_dataset,
                    cfg=cfg,
                )
            eval_results["policy_refinement"] = p + 1
            # Save evaluation results
            evaluation_results["max_reward"].append(eval_results["max_reward"])
            evaluation_results["success"].append(eval_results["success"])
            # @TODO remove after testing
            if cfg.evaluation.eval:
                with open(
                    os.path.join(out_dir, "eval_results.pkl"), "wb"
                ) as f:
                    print(
                        "Results saved at: ",
                        os.path.join(out_dir, "eval_results.pkl"),
                    )
                    pickle.dump(evaluation_results, f)
            log_eval_info(logger, eval_results, step, cfg, "eval")
            if cfg.wandb.enable:
                logger.log_image(
                    eval_results["last_goal"],
                    "goal_conditioning",
                    step,
                    mode="eval",
                )
                if (
                    cfg.evaluation.progressive_goals
                    and cfg.evaluation.save_progressive_goals
                ):
                    logger.log_image(
                        eval_results["starting_state"],
                        "starting_state",
                        step,
                        mode="eval",
                    )
                    logger.log_image(
                        eval_results["closest_expert"],
                        "closest_expert",
                        step,
                        mode="eval",
                    )
                    for i, x in enumerate(eval_results["goal_curriculum"]):
                        logger.log_image_locally(
                            np.moveaxis(x, 0, -1).astype(np.float32),
                            f"goal_curriculum_{p}_{i}",
                            "viz_curriculum",
                        )

                logger.log_video(
                    str(eval_results["rollout_video"]), step, mode="eval"
                )

    # Save evaluation results
    if cfg.evaluation.eval:
        with open(os.path.join(out_dir, "eval_results.pkl"), "wb") as f:
            pickle.dump(evaluation_results, f)


@hydra.main(
    version_base="1.2", config_path="../../config", config_name="config"
)
def train_cli(cfg: DictConfig) -> None:
    """Training from a configuration file."""
    training_config(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    train_cli()
