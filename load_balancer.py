import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class LoadBalancer(gym.Env):
    """
    grid simulation environment for reinforcement learning.

    this environment simulates a simplified smart grid with multiple energy sources and storage capabilities. It is designed to handle variable energy demand over time. An agent interacting with this environment must make decisions on energy production and storage to meet demand efficiently.
    
    attributes:
        action_space (spaces.MultiDiscrete): The space of possible actions an agent can take. Each action represents decisions on energy production levels and storage.
        observation_space (spaces.Box): The space of possible states the environment can be in. Each state includes the current energy demand and the amount of energy stored.
        state (np.array): The current state of the environment, including demand and storage levels.
        time_step (int): A counter tracking the number of steps taken in the environment.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(LoadBalancer, self).__init__()
        
        # optional, defined action/observation spaces
        # self.num_sources = 3  # energy sources
        # self.num_storages = 5  # energy storage units
        # self.num_consumers = 15  # energy consumers
        
        # define action/observation spaces
        self.action_space = spaces.MultiDiscrete([11, 11, 11, 21])
        self.observation_space = spaces.Box(low=np.array([0, 0], dtype=np.float32),
                                            high=np.array([20, 20], dtype=np.float32),
                                            dtype=np.float32)
        # initialize state/timestep
        self.state = np.array([10, 10], dtype=np.float32)
        self.time_step = 0
        
        # store results
        self.demand_history = []
        self.storage_history = []
        self.time_steps_history = []
        self.rewards_history = []

    def step(self, action):
        """
        updates the environment's state based on the agent's action

        parameters:
            action (np.array): action taken by agent, including decisions on energy production and storage
        
        returns:
            np.array: the new state of the environment after taking the action
            float: the reward received after taking the action
            bool: whether the episode has ended (always False in this environment)
            dict: additional information about the step (empty in this environment)
        """
        
        demand_pattern = 10 + 3 * np.sin(self.time_step * 0.3 * np.pi) + 2 * np.sin(self.time_step * 0.1 * np.pi)
        self.state[0] = demand_pattern + np.random.normal(0, 1.5) 
        self.time_step += 1
        
        total_production = sum(action[:-1])
        storage_action = action[-1]
        
        self.state[1] += total_production - self.state[0]
        self.state[1] = min(max(self.state[1], 0), 20)
        
        reward = -abs(total_production - self.state[0]) - abs(storage_action - self.state[1])
        done = False
        info = {}
        
        # store results
        self.demand_history.append(self.state[0])
        self.storage_history.append(self.state[1])
        self.time_steps_history.append(self.time_step)
        self.rewards_history.append(reward)
        
        return self.state, reward, done, info

    def reset(self):
        """
        resets environment to initial state

        returns:
            np.array: initial state of the environment
        """
        self.state = np.array([10, 10], dtype=np.float32)
        self.time_step = 0
        return self.state

    def render(self, mode='console'):
        """
        renders the current state of the environment

        parameters:
            mode (str): renders to console
        """
        if mode != 'console':
            raise NotImplementedError()
        print(f"Time Step: {self.time_step}, Demand: {self.state[0]}, Storage: {self.state[1]}")

    def seed(self, seed=None):
        """
        sets the seed for this environment's random number generator

        parameters:
            seed (int, optional): if none, random will be used
        """
        np.random.seed(seed)


# train
env = make_vec_env(lambda: LoadBalancer(), n_envs=1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("grid_optimization_model")
env.close()

# test
model = PPO.load("grid_optimization_model")
env = LoadBalancer()

obs = env.reset()
for i in range(50):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break
