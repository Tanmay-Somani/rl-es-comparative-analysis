import os
import argparse
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from grid import TreasureMazeEnv
from logs.utils import save_reward_plot, log_episode

def make_env():
    env = TreasureMazeEnv()
    return Monitor(env)

train_env = make_vec_env(make_env, n_envs=1)
model = PPO("MlpPolicy", train_env, verbose=1)


log_dir = "logs/centralized_rl"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "episode_rewards.csv")
episode_rewards = []


TIMESTEPS = 5000
EVAL_EPISODES = 100

print("Training centralized PPO agent...")
model.learn(total_timesteps=TIMESTEPS)

eval_env = TreasureMazeEnv()

print("🔍 Evaluating trained agent...")
for episode in tqdm(range(EVAL_EPISODES), desc="Evaluating"):
    obs, _ = eval_env.reset()
    obs = np.array([obs], dtype=np.float32)  
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs_raw, reward, done, _, _ = eval_env.step(action[0])
        obs = np.array([obs_raw], dtype=np.float32)
        total_reward += reward
    episode_rewards.append(total_reward)
    log_episode(log_file, episode, total_reward)


save_reward_plot(episode_rewards, filename=os.path.join(log_dir, "reward_plot.png"))
print(f" Finished centralized RL training. Plot saved to {log_dir}/reward_plot.png")
