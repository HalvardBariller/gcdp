"""This script trains a diffusion model for the PushT task."""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="transformers.deepspeed module is deprecated and will be removed in a future version")

import collections
import gymnasium as gym
import gym_pusht
import logging
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import tqdm

from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from gcdp.diffusion import (
    get_resnet,
    replace_bn_with_gn,
    ConditionalUnet1D,
)
from gcdp.episodes import (
    get_random_rollout,
    get_guided_rollout,
    PushTDatasetFromTrajectories,
)
from gcdp.logging import init_logging
from gcdp.utils import ScaleRewardWrapper, set_global_seed

# Create the environment with sparse rewards
env = gym.make(
    "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
)
env = ScaleRewardWrapper(env)

# Get statistics from demonstrations for normalization
# demonstrations = LeRobotDataset("lerobot/pusht")
# demonstrations_statistics = {
#     k: demonstrations.stats[k] for k in ["action", "observation.state"]
# }
demonstration = np.load(
    "objects/demonstration_statistics.npz",
    allow_pickle=True,
)
demonstration_statistics = {
    key: demonstration[key].item() for key in demonstration
}
demonstration.close()
# Load target goals
with open("objects/successes.pkl", "rb") as f:
    successes = pickle.load(f)
for item in successes:
    item["pixels"] = item["pixels"].astype(np.float64)


network_params = {
    "obs_horizon": 2,
    "pred_horizon": 16,
    "action_horizon": 8,
    "action_dim": 2,
    "num_diffusion_iters": 100,
    "batch_size": 64,
    "policy_refinement": 2,
    "num_epochs": 1,
    "episode_length": 50,
    "num_episodes": 10,
}

# Set seed for reproducibility
set_global_seed(42)
# Configure logging
init_logging()

# Prediction parameters
pred_horizon = network_params["pred_horizon"]
obs_horizon = network_params["obs_horizon"]
action_horizon = network_params["action_horizon"]
# Training parameters
policy_refinement = network_params["policy_refinement"]
num_epochs = network_params["num_epochs"]
batch_size = network_params["batch_size"]

episode_length = network_params["episode_length"]
num_episodes = network_params["num_episodes"]


# Networks definition
# construct ResNet18 encoder
vision_encoder = get_resnet("resnet18")
# replace all BatchNorm with GroupNorm to work with EMA
vision_encoder = replace_bn_with_gn(vision_encoder)
# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim  # 514
action_dim = network_params["action_dim"]  # 2
goal_dim = vision_feature_dim + lowdim_obs_dim  # 514
# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon + goal_dim
)  # gloabl_cond_dim = 514 * 2 + 514 = 1542

# the final arch has 2 parts
nets = nn.ModuleDict(
    {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
)
# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = network_params["num_diffusion_iters"]
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule="squaredcos_cap_v2",
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type="epsilon",
)
# device transfer
device = torch.device("cuda")
_ = nets.to(device)
# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(parameters=nets.parameters(), power=0.75)
# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4,
    weight_decay=1e-6,
    eps=1e-8,
    betas=(0.95, 0.999),
)

logging.info("Training Diffusion Model.")
for p in range(policy_refinement):
    if p == 0:
        trajectories = [
            get_random_rollout(episode_length=episode_length, env=env)
            for _ in range(num_episodes)
        ]
        # create dataset
        dataset = PushTDatasetFromTrajectories(
            trajectories,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            get_original_goal=False,
            dataset_statistics=demonstration_statistics,
        )
        # create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=False,
        )
        # Cosine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * num_epochs,
        )
    else:
        logging.info("Generating new trajectories...")
        trajectories = [
            get_guided_rollout(
                episode_length=episode_length,
                env=env,
                model=nets,
                device=device,
                network_params=network_params,
                normalization_stats=demonstration_statistics,
                noise_scheduler=noise_scheduler,
            )
            for _ in range(num_episodes)
        ]
        # update dataset
        dataset = PushTDatasetFromTrajectories(
            trajectories,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            get_original_goal=False,
            dataset_statistics=demonstration_statistics,
        )
        # update dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=False,
        )
        print("Dataloading OK")

    with tqdm.tqdm(range(num_epochs), desc="Epoch") as tglobal:
        # epoch loop
        for nepoch in tglobal:
            logging.info("EPOCH: %d", nepoch+1)
            epoch_loss = []
            # batch loop
            step = 0
            with tqdm.tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
                    step += 1
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch["image"][:, :obs_horizon].to(
                        device
                    )  # (B, obs_horizon, C, H, W) = (64, 2, 3, 96, 96)
                    nagent_pos = nbatch["agent_pos"][:, :obs_horizon].to(
                        device
                    )  # (B, obs_horizon, 2) = (64, 2, 2)
                    naction = nbatch["action"].to(
                        device
                    )  # (B, pred_horizon, 2) = (64, 16, 2)
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
                    image_features = nets["vision_encoder"](
                        nimage.flatten(end_dim=1)
                    )
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
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps
                    )

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=full_cond
                    )

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # Log the loss
                    log_items = [
                        f"Policy Refinement: {p+1}",
                        f"Epoch: {nepoch+1}",
                        f"Step: {step}",
                        f"Loss: {loss.item():.3f}",
                        # f"Learning Rate: {lr_scheduler.get_last_lr():.3f}",
                        f"Learning Rate: {optimizer.param_groups[0]['lr']:0.1e}"
                    ]
                    if step % 10 == 0:
                        logging.info(" | ".join(log_items))

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            # Log epoch loss
            log_epoch = [
                f"Policy Refinement: {p+1}",
                f"Epoch: {nepoch+1}",
                f"Epoch Loss: {np.mean(epoch_loss):.3f}",
                f"Learning Rate: {optimizer.param_groups[0]['lr']:0.1e}",
            ]
            logging.info(" | ".join(log_epoch))

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())

    # Evaluate the model
    max_steps = 200
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    env = ScaleRewardWrapper(env)
    # get first observation
    obs, info = env.reset()
    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards


# Save the model
torch.save(ema_nets, "objects/pusht_model.pth")

