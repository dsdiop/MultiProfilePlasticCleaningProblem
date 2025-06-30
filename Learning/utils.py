import gym
import numpy as np
import os
import torch
import torch.nn as nn
from gym import spaces

import sys
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.Expert_nu import Expert_nu
from typing import Dict
from torch.distributions import Bernoulli
import argparse

class MultiAgentNuWrapper(gym.Wrapper):
    # This wrapper will:

#     Accept nu values (continuous in [0, 1]) from the PPO agent.

#     Use Expert_nu to convert nu into discrete actions for each agent.

#     Manage the multi-agent-to-single-agent conversion for PPO.
    def __init__(self, env, expert_nu):
        super().__init__(env)
        self.expert_nu = expert_nu
        self.num_agents = env.number_of_agents
        
        # PPO interacts with nu values for all agents
        # self.single_action_space = spaces.Box(
        #     low=0.0, 
        #     high=1.0, 
        #     shape=(self.num_agents,),  # One nu per agent
        #     dtype=np.float32
        # )
        # For multi-agent discrete actions mu=0 or nu=1,  we use MultiBinary
        self.single_action_space = spaces.MultiBinary(self.num_agents)

        # ✅ Flattened observation space: assumes get_agent_state returns np.ndarray
        sample_obs = self._flatten_obs(env.reset()[0])
        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        # Initialize last coverage metrics
        self.last_trash_coverage = 0.0
        self.last_map_coverage = 0.0
        self.simulate_steps = False  # Flag to control simulation mode

    def reset(self):
        obs , _= self.env.reset()
        
        # Initialize last coverage metrics
        self.last_trash_coverage = 0.0
        self.last_map_coverage = 0.0
        return self._flatten_obs(obs) , _

    def step(self, action_nu: np.ndarray):
        """
        action_nu: Array of shape (num_agents,) with nu_i ∈ [0,1]
        """
        # Convert nu to actions using Expert_nu
        actions = {}
        state_float32 = {i:None for i in self.env.state.keys()}
        if self.env.convert_to_uint8:
            for agent_id in self.env.state.keys():
                state_float32[agent_id] = (self.env.state[agent_id] / 255.0).astype(np.float32)
        else:
            state_float32 = self.env.state
        # Create a condition array for each agent based on nu values
        condition = np.ndarray(shape=(self.env.number_of_agents,), dtype=bool)
        for agent_id in range(self.env.number_of_agents):
            condition[agent_id] = action_nu[agent_id] > np.random.rand()
        # condition = None
        # Select actions using Expert_nu
        if not self.expert_nu.masked_actions:
            actions = self.expert_nu.select_action(state_float32,condition=condition)
        else:
            actions = self.expert_nu.select_masked_action(states=state_float32, positions=self.env.fleet.get_positions(),condition=condition)

        actions = {agent_id: action for agent_id, action in actions.items()}
        # Process the agent step #
        # Step the environment
        if self.simulate_steps:
            # If simulating, we don't actually step the environment
            reward = self.env.simulate_step(actions)
            return None, np.sum(list(reward.values())) , None, None, {}
        next_obs, reward, done, info = self.env.step(actions)
        
        current_trash = info["trash_coverage"]  # e.g., 0.57
        current_map = info["map_coverage"]      # e.g., 0.62

        delta_trash = current_trash - self.last_trash_coverage
        delta_map = current_map - self.last_map_coverage

        self.last_trash_coverage = current_trash
        self.last_map_coverage = current_map

        reward = np.sum(list(reward.values())) 

        # For PPO: Flatten obs, average rewards, aggregate dones
        return self._flatten_obs(next_obs), reward, any(done.values()), any(done.values()), info
        

    def _get_agent_state(self, agent_id: int) -> np.ndarray:
        """Implement this method in your env to return agent-specific observations"""
        return self.env.get_agent_state(agent_id)

    def _flatten_obs(self, obs: Dict[int, np.ndarray]) -> np.ndarray:
        # return np.concatenate([obs[agent_id] for agent_id in sorted(obs.keys())])
        """Stack all agent observations into a single vector"""
        # We want: [trash_map, egopos_0, otherpos_0, egopos_1, otherpos_1, ..., egopos_N, otherpos_N]
        # Assume trash_map is the same for all agents, so take from agent 0
        trash_map = obs[0][0:1]  # shape (1, H, W) or (1, ...)
        fleet_obs = [trash_map]
        for agent_id in sorted(obs.keys()):
            fleet_obs.append(obs[agent_id][1:])  # egopos_i, otherpos_i
        # fleet_obs = np.concatenate(fleet_obs, axis=0)
        # obs = {0: fleet_obs}
        # return np.concatenate([obs[agent_id] for agent_id in sorted(obs.keys())])
        # Flatten the observations
        return np.concatenate(fleet_obs, axis=0)  # shape (num_agents * obs_dim,)
        
        # return obs

def make_env(
    env_fn: callable,  # Function that returns a MultiAgentPatrolling instance
    expert_nu_fn: callable,  # Function that returns an Expert_nu instance
    env_kwargs: dict,  # Arguments for MultiAgentPatrolling
    expert_kwargs: dict,  # Arguments for Expert_nu
    seed: int,
    device_int: int,  # Device for PyTorch
    args: argparse.Namespace,  # Arguments from the command line
    idx: int,
    capture_video: bool = False,
    run_name: str = None,
):
    def thunk():
        # Initialize environment with custom args
        env = env_fn(**env_kwargs)
        # path planner is a DQFDuelingVisualNetwork
        from Algorithm.RainbowDQL.Networks.network import DQFDuelingVisualNetwork
        nettype = '0'
        arch = 'v1'
        device = 'cpu' if device_int == -1 else f'cuda:{device_int}'
        path_planner = DQFDuelingVisualNetwork(env.observation_space.shape, [8, env.action_space.n - 8], 1024,arch,nettype).to(device)
        # Experimento_clean26_alamillo_lake_macro_plastic_random_nus_nsteps5 Experimento_clean26_malaga_port_macro_plastic_random_nus_nsteps5
        
        path_to_file = f"{data_path}/Learning/path_planner_algorithms/Experimento_clean26_{args.map}_{args.benchmark}_random_nus_nsteps5/Final_Policy.pth"
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"Path to file {path_to_file} does not exist. Please check the path.")
        
        path_planner.load_state_dict(torch.load(path_to_file, map_location=device))
        ########################
        
        # Initialize Expert_nu with env instance + custom args
        expert_nu = expert_nu_fn(env=env, path_planner=path_planner, **expert_kwargs)
        
        # Wrap the environment
        env = MultiAgentNuWrapper(env, expert_nu)
        
        # Standard Gym wrappers
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video and idx == 0:
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        # Seeding
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(envs.single_observation_space.shape[0], 32, 8, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        n_flatten = np.prod(self.network(torch.zeros(size=(1,
                                                               envs.single_observation_space.shape[0],
                                                               envs.single_observation_space.shape[1],
                                                               envs.single_observation_space.shape[2]))).shape)
        self.linear = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU(),
        )
        n_actions = envs.single_action_space.n if isinstance(envs.single_action_space, spaces.MultiBinary) else envs.single_action_space.shape[0]
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)

        # self.actor = nn.Sequential(
        #                             layer_init(nn.Linear(512, envs.single_action_space.shape[0]), std=0.01),
        #                             nn.Sigmoid()  # To ensure outputs ∈ [0,1])
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = self.network(x)
        x = self.linear(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.network(x)
        x = self.linear(x)
        
        logits = self.actor(x)  # shape: (batch, n_agents)
        dist = Bernoulli(logits=logits)  # or probs=torch.sigmoid(logits)

        if action is None:
            action = dist.sample()
        
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, logprob, entropy, self.critic(x)

import itertools
import numpy as np
import copy
from gym.vector import SyncVectorEnv

# Suppose you want to evaluate C candidates each step
# candidate_count = C
# vec_env = SyncVectorEnv([make_env for _ in range(candidate_count)])
# obs_batch = vec_env.reset()
class GreedyAgent:
    def __init__(self, env):
        self.num_agents = env.num_agents        
        self.env = env
        import time
        # print the time it takes to copy the environment
        # start_time = time.time()
        # self.sim_env = copy.deepcopy(env)
        # end_time = time.time()
        # print(f"Time taken to copy the environment: {end_time - start_time:.4f} seconds")
        # start_time = time.time()
        # _ = copy.deepcopy(env.fleet)
        # end_time = time.time()
        # print(f"Time taken to copy the environment: {end_time - start_time:.4f} seconds")
        # start_time = time.time()
        # self.sim_env = copy.deepcopy(env.gt)
        # end_time = time.time()
        # print(f"Time taken to copy the environment: {end_time - start_time:.4f} seconds")
        # self.sim_env.reset()
        # Ensure sim_env is a SyncVectorEnv for batch processing
        # if not isinstance(self.sim_env, SyncVectorEnv):
        #     self.sim_env = SyncVectorEnv([lambda: self.sim_env for _ in range(1)])

    def act(self, obs):
        best_nu = None
        best_reward = -float("inf")
        self.env.simulate_steps = True
        for nu in itertools.product([0, 1], repeat=self.num_agents):
            nu_vec = np.array(nu, dtype=np.float32)

            _, reward , _, _, _ = self.env.step(nu_vec)

            if reward > best_reward:
                best_reward = reward
                best_nu = nu_vec
        self.env.simulate_steps = False
        return best_nu

