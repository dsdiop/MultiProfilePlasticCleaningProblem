
# Train a PPO agent to solve the multiobjective cleaning problem.
# This script sets up a multi-agent environment, defines a PPO agent, and trains it using the Proximal Policy Optimization algorithm.

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TrashMapEncoder(nn.Module):
#     def __init__(self, output_dim=64):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 4)),
#             nn.Flatten(),
#             nn.Linear(32 * 4 * 4, output_dim),
#             nn.ReLU(),
#         )

#     def forward(self, trash_map):  # [B, 1, H, W]
#         return self.encoder(trash_map)  # [B, D]


# class AgentEncoder(nn.Module):
#     def __init__(self, input_channels=2, trash_dim=64, hidden_dim=128):
#         super().__init__()
#         self.agent_cnn = nn.Sequential(
#             nn.Conv2d(input_channels, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 4)),
#             nn.Flatten(),
#             nn.Linear(32 * 4 * 4, hidden_dim),
#             nn.ReLU()
#         )
#         self.nu_head = nn.Sequential(
#             nn.Linear(hidden_dim + trash_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),  # Single nu
#             nn.Sigmoid()
#         )

#     def forward(self, ego_other_map, trash_feat):
#         x = self.agent_cnn(ego_other_map)         # [B, hidden_dim]
#         x = torch.cat([x, trash_feat], dim=-1)    # concat shared map
#         nu = self.nu_head(x).squeeze(-1)          # [B]
#         return nu, x  # return intermediate features too


# class PPOAgent(nn.Module):
#     def __init__(self, num_agents, trash_map_shape=(1, 64, 64), ego_other_shape=(2, 64, 64),
#                  trash_dim=64, hidden_dim=128):
#         super().__init__()
#         self.num_agents = num_agents
#         self.trash_encoder = TrashMapEncoder(output_dim=trash_dim)
#         self.agent_encoder = AgentEncoder(input_channels=2, trash_dim=trash_dim, hidden_dim=hidden_dim)
#         self.critic_head = nn.Sequential(
#             nn.Linear((hidden_dim + trash_dim) * num_agents, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, trash_map, agent_maps):
#         """
#         trash_map: Tensor of shape [B, 1, H, W] (shared map)
#         agent_maps: list of length num_agents, each is [B, 2, H, W] (ego + others per agent)
#         """
#         batch_size = trash_map.shape[0]
#         trash_feat = self.trash_encoder(trash_map)  # [B, D]

#         nu_list = []
#         feat_list = []
#         for agent_idx in range(self.num_agents):
#             agent_map = agent_maps[agent_idx]  # [B, 2, H, W]
#             nu_i, feat_i = self.agent_encoder(agent_map, trash_feat)  # [B], [B, D+H]
#             nu_list.append(nu_i.unsqueeze(1))
#             feat_list.append(feat_i)

#         self.last_feats = torch.cat(feat_list, dim=-1)  # store for critic
#         nu = torch.cat(nu_list, dim=1)  # [B, num_agents]
#         return nu

#     def get_value(self):
#         return self.critic_head(self.last_feats)  # [B, 1]

#     def get_action_and_value(self, trash_map, agent_maps):
#         nu = self.forward(trash_map, agent_maps)
#         value = self.get_value()
#         return nu, value


# #33333333333333333


import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym import spaces
from typing import Dict, Tuple
import sys
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)

from Learning.utils import make_env, Agent
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.Expert_nu import Expert_nu
from gym.spaces import MultiBinary
from torch.distributions import Bernoulli
from tqdm import trange


## Code adapted from the 'ppo_atari.py' example in the following repository:
## https://github.com/vwxyzjn/ppo-implementation-details

def parse_args():
    parser = argparse.ArgumentParser('Train a PPO agent to solve the multiobjective cleaning problem.')
    # Environment specific arguments
    parser.add_argument('--map', type=str, default='malaga_port', choices=['malaga_port','alamillo_lake','ypacarai_map'], 
        help='The map to use.')
    parser.add_argument('--distance_budget', type=int, default=200, 
        help='The maximum distance of the agents.')
    parser.add_argument('--n_agents', type=int, default=4, 
        help='The number of agents to use.')
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument('--miopic', type=bool, default=True, 
        help='If True the scenario is miopic.')
    parser.add_argument('--detection_length', type=int, default=2, 
        help='The influence radius of the agents.')
    parser.add_argument('--movement_length', type=int, default=1, 
        help='The movement length of the agents.')
    parser.add_argument('--reward_type', type=str, default='Distance Field', 
        help='The reward type to train the agent.')
    parser.add_argument('--convert_to_uint8', type=bool, default=False, 
        help='If convert the state to unit8 to store it (to save memory).')
    parser.add_argument('--benchmark', type=str, default='macro_plastic', choices=['shekel', 'algae_bloom','macro_plastic'], 
        help='The benchmark to use.')
    parser.add_argument('--dynamic', type=bool, default=True, 
        help='Simulate dynamic')
    parser.add_argument('--device', type=int, default=0, help='The device to use.', choices=[-1, 0, 1])
    
    # Path planner specific arguments
    parser.add_argument('--path-planner-model', type=str, default='vaeUnet', choices=['miopic', 'vaeUnet'], 
        help='The model to use.')

    # fmt: off
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.map}__{args.benchmark}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import json

    # Save all command-line arguments to a JSON file
    with open(f"runs/{run_name}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    print("Run name:", run_name)
    print("✅ Arguments saved to args.json")
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device_str = 'cpu' if args.device == -1 else f'cuda:{args.device}'
    device = torch.device("cuda" if torch.cuda.is_available() and device_str else "cpu")
    print(f"\nUsing device: {device}\n")
    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )
    
    # Environment setup
    N = args.n_agents
    sc_map = np.genfromtxt(f"{data_path}/Environment/Maps/{args.map}.csv", delimiter=',')

    if args.map == 'malaga_port':
        initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
    elif args.map == 'alamillo_lake':
        initial_positions = np.array([[68, 26], [64, 26], [60, 26], [56, 26]])[:N, :]
    elif args.map == 'ypacarai_map':
        initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])


    env_kwargs = {
        "scenario_map": sc_map,
        "fleet_initial_positions": initial_positions,
        "distance_budget": args.distance_budget,
        "number_of_vehicles": N,  # Or use args.num_agents if defined
        "seed": args.seed,
        "miopic": args.miopic,
        "dynamic": args.dynamic,
        "detection_length": args.detection_length,
        "movement_length": args.movement_length,
        "max_collisions": 15,
        "reward_type": args.reward_type,
        "convert_to_uint8": args.convert_to_uint8,
        "ground_truth_type": args.benchmark,
        "obstacles": False,
        "frame_stacking": 1
    }

    expert_kwargs = {
    # "env": env, # environment will be passed in the thunk
    "device": device_str,
    "masked_actions": True,
    "consensus": True,
    }


    env_fns = [
        make_env(
            env_fn=MultiAgentPatrolling,
            expert_nu_fn=Expert_nu,
            env_kwargs=env_kwargs,
            expert_kwargs=expert_kwargs,
            seed=args.seed + i,
            device_int=args.device,
            args=args,
            idx=i,
            capture_video=(i == 0),
            run_name=run_name
        ) for i in range(args.num_envs)
    ]

    envs = gym.vector.SyncVectorEnv(env_fns)
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only box action discrete is supported"
    # For shape extraction only; agent will work with envs later
    env_for_shapes = env_fns[0]()
    agent = Agent(env_for_shapes).to(device)

    # agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in trange(1, num_updates + 1, desc="Updates"):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        total_reward = 0.0
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            if global_step==1176:
                print(f"global_step={global_step}, action={action.cpu().numpy()}")
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated) # because gym >= 0.26 returns both terminated and truncated
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            total_reward += np.mean(reward)
            
            # for item in info:
            #     if "episode" in item:
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break

            # for item in info:
            #     if "episode" in item.keys():
            #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("\nSteps", int(global_step), "SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # record rewards for plotting purposes
        writer.add_scalar("charts/mean_reward", total_reward, global_step) # Mean reward across all environments

    envs.close()
    writer.close()
    # Save the trained PPO agent
    torch.save(agent.state_dict(), f"runs/{run_name}/ppo_agent.pth")
    print("✅ Agent saved to", f"ppo_agent.pth")