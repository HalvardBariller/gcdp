"""Evaluate the model on the moving target environment."""

import collections
import gymnasium as gym
import hydra
import os
import numpy as np
import torch
import tqdm

from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
from torch import nn

from gcdp.model.modeling import make_diffusion_model
from gcdp.model.policy import diff_policy
from gcdp.scripts.common.utils import (
    get_demonstration_statistics,
    set_global_seed,
)
from gcdp.scripts.training.expert_train import get_demonstration_successes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(cfg: DictConfig) -> nn.Module:
    """Build the model and load the weights."""
    script_dir = Path().resolve()
    model_dir = (
        script_dir
        / "runs/outputs/train/2024-09-06/16-30-06_evengoals_wo_curr/diffusion_model.pth"
    )
    nets, ema_nets, _, noise_scheduler = make_diffusion_model(cfg)
    model = torch.load(model_dir, map_location=device)
    ema_nets.load_state_dict(model)
    return ema_nets, noise_scheduler


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


def eval_policy(
    env,
    dummy_env,
    success_path,
    num_episodes,
    max_steps,
    save_video,
    network_params,
    seed,
    model,
    noise_scheduler,
    device,
    normalization_stats,
    video_path,
    video_prefix,
    cfg: DictConfig,
    progressive_goals=False,
) -> None:
    """Evaluate the model."""
    actions_taken = network_params["action_horizon"]
    obs_horizon = network_params["obs_horizon"]
    episode_results = {
        "success": [],
        "rewards": [],
        "max_reward": [],
        "goal_curriculum": [],
    }
    for episode in tqdm.tqdm(range(num_episodes)):
        if save_video and episode == num_episodes - 1:
            env.metadata["render_fps"] = 10
            env = RecordVideo(
                env, video_path, disable_logger=True, name_prefix=video_prefix
            )
            print("FPS", env.metadata["render_fps"])
        seed += 1
        task_completed = False
        # Keep track of the planned actions
        action_queue = collections.deque(maxlen=actions_taken)
        # Initialize the environment
        s, _ = env.reset(seed=seed)
        goal_pose = env.get_wrapper_attr("goal_pose")
        # Get the desired goal
        desired_goal, _ = dummy_env.reset(target_coordinates=goal_pose)
        dummy_env.close()
        desired_goal = {"pixels": desired_goal["pixels"] / 255.0}
        # designed_success = get_demonstration_successes(
        #     success_path
        # )
        # desired_goal = designed_success[0]
        episode_results["starting_state"] = s["pixels"]
        done = False
        tot_reward = 0
        max_reward = 0
        observations = collections.deque([s] * obs_horizon, maxlen=obs_horizon)
        step = 0

        while not done:
            # Execute the planned actions
            if action_queue:
                s, r, done, _, _ = env.step(action_queue.popleft())
                tot_reward += r
                max_reward = max(max_reward, r)
                step += 1
                if done:
                    task_completed = True
                # Update the observations
                observations.append(s)
            # Plan new actions
            else:
                behavioral_goal = desired_goal
                goal_preprocessed = False
                action_chunk = diff_policy(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    observations=observations,
                    goal=behavioral_goal,
                    device=device,
                    network_params=network_params,
                    normalization_stats=normalization_stats,
                    actions_taken=actions_taken,
                    goal_preprocessed=goal_preprocessed,
                )
                action_queue.extend(action_chunk)
            if step > max_steps:
                done = True
        episode_results["success"].append(task_completed)
        episode_results["rewards"].append(tot_reward)
        episode_results["max_reward"].append(max_reward)

    if save_video and episode == num_episodes - 1:
        saved_path = env.video_recorder.path
        relative_video_path = os.path.relpath(saved_path)
        episode_results["rollout_video"] = relative_video_path
    env.close()
    episode_results["sum_rewards"] = sum(episode_results["rewards"])
    episode_results["success_rate"] = (
        sum(episode_results["success"]) / num_episodes
    )
    episode_results["average_reward"] = (
        episode_results["sum_rewards"] / num_episodes
    )
    episode_results["average_max_reward"] = (
        sum(episode_results["max_reward"]) / num_episodes
    )
    episode_results["last_goal"] = desired_goal["pixels"]

    return episode_results


def eval_moving_target(
    cfg: DictConfig,
    out_dir: str,
    job_name: str,
) -> None:
    """Evaluate on moving targets."""
    set_global_seed(cfg.seed)
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        random_target=True,
    )
    dummy_env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        random_target=True,
        legacy=False,
    )
    ema_nets, noise_scheduler = load_model(cfg)
    network_params = build_params(cfg)
    stats_path = Path().resolve() / "objects/demonstration_statistics.npz"
    success_path = Path().resolve() / "objects/designed_success.pkl"
    demonstration_statistics = get_demonstration_statistics(path=stats_path)
    video_path = "video/pusht/" + job_name
    video_prefix = "pusht_changing_goal"
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.use_amp):
        eval_results = eval_policy(
            env=env,
            dummy_env=dummy_env,
            success_path=success_path,
            num_episodes=cfg.eval_generalisation.num_episodes,
            max_steps=cfg.eval_generalisation.max_steps,
            save_video=cfg.eval_generalisation.save_video,
            network_params=network_params,
            seed=cfg.seed,
            model=ema_nets,
            noise_scheduler=noise_scheduler,
            device=cfg.device,
            normalization_stats=demonstration_statistics,
            video_path=video_path,
            video_prefix=video_prefix,
            cfg=cfg,
        )
        # Save the image eval_results["last_goal"] to the output directory
        last_goal = np.array(eval_results["last_goal"])
        last_goal = (last_goal * 255).astype(np.uint8)
        last_goal = Image.fromarray(last_goal)
        last_goal.save(os.path.join(out_dir, "last_goal.png"))


@hydra.main(
    version_base="1.2", config_path="../../config", config_name="config"
)
def eval_cli(cfg: DictConfig) -> None:
    """CLI for evaluation."""
    eval_moving_target(
        cfg=cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    eval_cli()
