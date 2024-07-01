

import collections
import gym
import numpy as np
from policy import diff_policy

def eval(
    env: gym.Env,
    num_episodes: int,
    max_steps: int,
    save_video: bool = False,
    video_path: str = None,
    video_fps: int = 30,
    seed: int = 42,
    verbose: bool = True,
    **kwargs,
):
    """
    Evaluate the current policy on a given environment using the diff_policy function.

    Parameters
    """
    model = kwargs["model"]
    noise_scheduler = kwargs["noise_scheduler"]
    observations = kwargs["observations"]
    device = kwargs["device"]
    network_params = kwargs["network_params"]
    normalization_stats = kwargs["normalization_stats"]
    actions_taken = network_params["action_horizon"]
    obs_horizon = network_params["obs_horizon"]
    
    successes = kwargs["successes"]
    len_successes = len(successes)

    episode_results = {
        "success": [],
        "rewards": [],
    }

    for episode in range(num_episodes):
        seed += 1
        env.seed(seed)
        # Keep track of the planned actions
        action_queue = collections.deque(maxlen=actions_taken)
        # Randomly select a goal among the successful ones
        goal = successes[np.random.randint(len_successes)]
        # Initialize the environment
        s, _ = env.reset()
        done = False
        tot_reward = 0
        observations = collections.deque([s] * obs_horizon, maxlen=obs_horizon)
        step = 0
        while not done:
            # Execute the planned actions
            if action_queue:
                s, r, done, _, _ = env.step(action_queue.popleft())
                tot_reward += r
                step += 1
            # Plan new actions
            else:
                action_chunk = diff_policy(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    observations=observations,
                    goal=goal,
                    device=device,
                    network_params=network_params,
                    normalization_stats=normalization_stats,
                    actions_taken=actions_taken,
                )
                action_queue.extend(action_chunk)
            # Update the observations
            observations.append(s)
            if step > max_steps:
                done = True
        episode_results["success"].append(done)
        episode_results["rewards"].append(tot_reward)
    episode_results["success_rate"] = sum(episode_results["success"]) / num_episodes
    episode_results["average_reward"] = sum(episode_results["rewards"]) / num_episodes
    return episode_results

                


# NEED TO ADD VIDEO RECORDING AND LOG EVERYTHING INTO WANDB


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
