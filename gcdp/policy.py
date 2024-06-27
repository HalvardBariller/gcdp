import collections
import gymnasium as gym
import gym_pusht
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from gcdp.episodes import normalize_data, unnormalize_data
from gcdp.utils import ScaleRewardWrapper


from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from gcdp.episodes import (
    get_rollout,
    split_trajectory,
    PushTDatasetFromTrajectories,
)
from gcdp.utils import ScaleRewardWrapper, record_video
from gcdp.diffusion import (
    get_resnet,
    replace_bn_with_gn,
    ConditionalUnet1D,
)


def diff_policy(
    model: torch.ModuleDict,
    observations: collections.deque,
    goal: np.ndarray,
    device: torch.device,
    network_params: dict,
    normalization_stats: dict,
):
    """
    Predict the action to take to reach the goal considering past observations.
    Inputs:
    - model: the model used to predict the action (vision and noise models)
    - observations: the past observations
    - goal: the goal to reach
    - device: the device to use
    - network_params: the parameters of the network
    - normalization_stats: the statistics used to normalize the data
    Outputs:
    - action: the action to take
    """
    # Unpack network parameters
    obs_horizon = network_params["obs_horizon"]
    pred_horizon = network_params["pred_horizon"]
    action_dim = network_params["action_dim"]
    num_diffusion_iters = network_params["num_diffusion_iters"]
    B = network_params["batch_size"]
    noise_scheduler = network_params["noise_scheduler"]

    images = np.stack([x["pixels"] for x in observations])
    agent_poses = np.stack([x["agent_pos"] for x in observations])
    # Normalization
    images /= 255.0
    agent_poses = normalize_data(
        agent_poses,
        stats=normalization_stats["agent_pos"],
    )

    # device transfer
    images = torch.from_numpy(images).to(device, dtype=torch.float32)
    # (2,3,96,96)
    agent_poses = torch.from_numpy(agent_poses).to(device, dtype=torch.float32)
    # (2,2)

    # infer action
    with torch.no_grad():
        # get image features
        image_features = model["vision_encoder"](images)
        # (2,512)

        # concat with low-dim observations
        obs_features = torch.cat([image_features, agent_poses], dim=-1)

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        # initialize action from Gaussian noise
        noisy_action = torch.randn(
            (B, pred_horizon, action_dim),
            device=device,
        )
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = model["noise_pred_net"](
                sample=naction, timestep=k, global_cond=obs_cond
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

    return action_pred[start, :]
