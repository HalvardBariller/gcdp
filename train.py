import collections
import gymnasium as gym
import gym_pusht
from gymnasium.wrappers import RecordVideo
from IPython.display import Video, Image
import matplotlib.pyplot as plt
import numba
import numpy as np
import torch
import torch.nn as nn
import tqdm

# from tqdm import tqdm
from copy import deepcopy
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from gcdp.episodes import (
    get_random_rollout,
    get_guided_rollout,
    split_trajectory,
    PushTDatasetFromTrajectories,
)
from gcdp.utils import ScaleRewardWrapper, record_video
from gcdp.diffusion import (
    get_resnet,
    replace_bn_with_gn,
    ConditionalUnet1D,
)

### Create the environment with sparse rewards
env = gym.make(
    "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
)
env = ScaleRewardWrapper(env)

### Get statistics from demonstrations for normalization
# demonstrations = LeRobotDataset("lerobot/pusht")
# demonstrations_statistics = {
#     k: demonstrations.stats[k] for k in ["action", "observation.state"]
# }
demonstration = np.load("objects/demonstration_statistics.npz", allow_pickle=True)
demonstration_statistics = {key: demonstration[key].item() for key in demonstration}
demonstration.close()

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

### Prediction parameters
pred_horizon = network_params["pred_horizon"]
obs_horizon = network_params["obs_horizon"]
action_horizon = network_params["action_horizon"]
### Training parameters
policy_refinement = network_params["policy_refinement"]
num_epochs = network_params["num_epochs"]
batch_size = network_params["batch_size"]

episode_length = network_params["episode_length"]
num_episodes = network_params["num_episodes"]


### Networks definition
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
        trajectories = [
            get_guided_rollout(
                episode_length=episode_length,
                env=env,
                model=nets,
                device=device,
                network_params=network_params,
                normalization_stats=demonstration_statistics,
                noise_scheduler=noise_scheduler,
            ) for _ in range(num_episodes)
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
    
    # training loop
    print(f"Policy Refinement {p}")
    with tqdm.tqdm(range(num_epochs), desc="Epoch") as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm.tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
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
                    B = nagent_pos.shape[0]  # Batch = 64

                    # encoder vision features
                    image_features = nets["vision_encoder"](
                        nimage.flatten(end_dim=1)
                    )  # (B*obs_horizon, C, H, W) = (128, 3, 96, 96) --> (128, 512)
                    image_features = image_features.reshape(
                        *nimage.shape[:2], -1
                    )  # (B, obs_horizon, D) = (64, 2, 512)

                    # encoder vision goal
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
                        0, noise_scheduler.config.num_train_timesteps, (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=full_cond
                    )

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # Log the loss
                    # print(loss)

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

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())


###############################################
###############################################
# # Need to adapt the code for inference on the PushTImageEnv


# # limit enviornment interaction to 200 steps before termination
# max_steps = 200
# env = gym.make(
#     "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
# )
# env = ScaleRewardWrapper(env)

# # get first observation
# obs, info = env.reset()

# # keep a queue of last 2 steps of observations
# obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
# # save visualization and rewards
# imgs = [env.render(mode="rgb_array")]
# rewards = list()
# done = False
# step_idx = 0


# with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
#     while not done:
#         B = 1
#         # stack the last obs_horizon number of observations
#         images = np.stack([x["image"] for x in obs_deque])
#         agent_poses = np.stack([x["agent_pos"] for x in obs_deque])

#         # # normalize observation
#         # nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
#         # images are already normalized to [0,1]
#         nimages = images

#         # device transfer
#         nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
#         # (2,3,96,96)
#         nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
#         # (2,2)

#         # infer action
#         with torch.no_grad():
#             # get image features
#             image_features = ema_nets["vision_encoder"](nimages)
#             # (2,512)

#             # concat with low-dim observations
#             obs_features = torch.cat([image_features, nagent_poses], dim=-1)

#             # reshape observation to (B,obs_horizon*obs_dim)
#             obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

#             # initialize action from Guassian noise
#             noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
#             naction = noisy_action

#             # init scheduler
#             noise_scheduler.set_timesteps(num_diffusion_iters)

#             for k in noise_scheduler.timesteps:
#                 # predict noise
#                 noise_pred = ema_nets["noise_pred_net"](
#                     sample=naction, timestep=k, global_cond=obs_cond
#                 )

#                 # inverse diffusion step (remove noise)
#                 naction = noise_scheduler.step(
#                     model_output=noise_pred, timestep=k, sample=naction
#                 ).prev_sample

#         # unnormalize action
#         naction = naction.detach().to("cpu").numpy()
#         # (B, pred_horizon, action_dim)
#         naction = naction[0]
#         action_pred = unnormalize_data(naction, stats=stats["action"])

#         # only take action_horizon number of actions
#         start = obs_horizon - 1
#         end = start + action_horizon
#         action = action_pred[start:end, :]
#         # (action_horizon, action_dim)

#         # execute action_horizon number of steps
#         # without replanning
#         for i in range(len(action)):
#             # stepping env
#             obs, reward, done, _, info = env.step(action[i])
#             # save observations
#             obs_deque.append(obs)
#             # and reward/vis
#             rewards.append(reward)
#             imgs.append(env.render(mode="rgb_array"))

#             # update progress bar
#             step_idx += 1
#             pbar.update(1)
#             pbar.set_postfix(reward=reward)
#             if step_idx > max_steps:
#                 done = True
#             if done:
#                 break

# # print out the maximum target coverage
# print("Score: ", max(rewards))

# # visualize
# from IPython.display import Video

# vwrite("vis.mp4", imgs)
# Video("vis.mp4", embed=True, width=256, height=256)
