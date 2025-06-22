import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from grid import TreasureMazeEnv

env = TreasureMazeEnv()
check_env(env)

from stable_baselines3.common.env_util import make_vec_env
vec_env = make_vec_env(lambda: TreasureMazeEnv(), n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)
