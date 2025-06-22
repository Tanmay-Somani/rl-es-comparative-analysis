import os
import numpy as np
from grid import TreasureMazeEnv
from policy import SimpleMLPPolicy
from logs.utils import log_episode, save_reward_plot

# Hyperparameters
EPISODES = 100
POPULATION_SIZE = 50
SIGMA = 0.2
LEARNING_RATE = 0.03
EPISODE_LENGTH = 100

log_dir = "logs/centralized_es"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "episode_rewards.csv")

env = TreasureMazeEnv()
input_dim = 2  # obs = [x, y]
output_dim = env.action_space.n
policy = SimpleMLPPolicy(input_dim, output_dim)
theta = policy.get_flat()  # Flattened weights

def evaluate(weights):
    policy.set_flat(weights)
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(EPISODE_LENGTH):
        action = policy.act(obs, policy.params)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

episode_rewards = []

for episode in range(EPISODES):
    noise = [np.random.randn(theta.size) for _ in range(POPULATION_SIZE)]
    rewards = [evaluate(theta + SIGMA * eps) for eps in noise]

    rewards = np.array(rewards)
    if np.std(rewards) > 1e-6:
        A = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
    else:
        A = rewards - np.mean(rewards)

    update = sum(A[i] * noise[i] for i in range(POPULATION_SIZE))
    theta += LEARNING_RATE / (POPULATION_SIZE * SIGMA) * update

    mean_reward = evaluate(theta)
    episode_rewards.append(mean_reward)
    log_episode(log_file, episode, mean_reward)
    print(f"Episode {episode+1}: Avg Reward = {mean_reward:.2f}")

save_reward_plot(episode_rewards, filename=os.path.join(log_dir, "reward_plot.png"))
print(f"Centralized ES completed. Logs and plot saved to {log_dir}")
