import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from grid import TreasureMazeEnv
from logs.utils import log_episode, save_reward_plot

NUM_AGENTS = 3
ROUNDS = 20
LOCAL_TIMESTEPS = 1000
EVAL_EPISODES = 50

log_dir = "logs/federated_rl"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "episode_rewards.csv")

def personalized_layout(index):
    layout = np.zeros((5, 5), dtype=np.uint8)
    layout[1 + index % 3, 2] = 1  # punishment varies
    layout[2, 1 + index % 3] = 2  # reward varies
    return layout

def make_personal_env(agent_id):
    env = TreasureMazeEnv(custom_layout=personalized_layout(agent_id))
    return Monitor(env)

# Create initial agents
agents = [
    PPO("MlpPolicy", make_vec_env(lambda: make_personal_env(i), n_envs=1), verbose=0)
    for i in range(NUM_AGENTS)
]

def average_weights(models):
    new_params = models[0].get_parameters()
    for key in new_params.keys():
        new_params[key] = sum(model.get_parameters()[key] for model in models) / len(models)
    return new_params

# Training loop
episode_rewards = []

for rnd in tqdm(range(ROUNDS), desc="Federated Rounds"):
    print(f"\n🌐 Round {rnd+1}/{ROUNDS}")

    # Local training
    for agent in agents:
        agent.learn(total_timesteps=LOCAL_TIMESTEPS, reset_num_timesteps=False)

    # Federated averaging
    averaged_params = average_weights(agents)
    for agent in agents:
        agent.set_parameters(averaged_params)

    # Evaluate averaged model on shared env
    eval_env = TreasureMazeEnv()
    total_eval_reward = 0
    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = agents[0].predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            total_eval_reward += reward

    avg_reward = total_eval_reward / EVAL_EPISODES
    log_episode(log_file, rnd, avg_reward)
    episode_rewards.append(avg_reward)
    print(f"✅ Avg Eval Reward: {avg_reward:.2f}")

# Save plot
save_reward_plot(episode_rewards, filename=os.path.join(log_dir, "reward_plot.png"))
print(f"\n🎉 Federated RL completed. Logs and plot saved to {log_dir}/reward_plot.png")
