import os
import numpy as np
from grid import TreasureMazeEnv
from policy import SimpleMLPPolicy
from logs.utils import log_episode, save_reward_plot

NUM_AGENTS = 3
ROUNDS = 100
LOCAL_POP = 20
SIGMA = 0.1
LEARNING_RATE = 0.02
LOCAL_ROUNDS = 5
MOMENTUM = 0.2
EPISODE_LENGTH = 50

log_dir = "logs/federated_es"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "episode_rewards.csv")

envs = [TreasureMazeEnv(custom_layout=None) for _ in range(NUM_AGENTS)]
input_dim = 2  
output_dim = envs[0].action_space.n
policies = [SimpleMLPPolicy(input_dim, output_dim) for _ in range(NUM_AGENTS)]
flat_weights = [p.get_flat() for p in policies]

def evaluate(env, flat_weights, policy, render=False):
    policy.set_flat(flat_weights)
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(EPISODE_LENGTH):
        if render:
            env.render()
        action = policy.act(obs, policy.params)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

episode_rewards = []

for rnd in range(ROUNDS):
    print(f"\n Round {rnd+1}/{ROUNDS}")
    new_weights = []

    for agent_id in range(NUM_AGENTS):
        policy = policies[agent_id]
        env = envs[agent_id]
        theta = flat_weights[agent_id]

        for local_round in range(LOCAL_ROUNDS):
            noise = [np.random.randn(theta.size) for _ in range(LOCAL_POP)]
            rewards = []

            for eps in noise:
                test_weights = theta + SIGMA * eps
                reward = evaluate(env, test_weights, policy)
                rewards.append(reward)

            rewards = np.array(rewards)
            if np.std(rewards) > 1e-6:
                A = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            else:
                A = rewards - np.mean(rewards)

            update = sum(A[i] * noise[i] for i in range(LOCAL_POP))
            theta += LEARNING_RATE / (LOCAL_POP * SIGMA) * update

        new_weights.append(theta)

    avg_weights = sum(new_weights) / NUM_AGENTS
    for i in range(NUM_AGENTS):
        flat_weights[i] = (1 - MOMENTUM) * flat_weights[i] + MOMENTUM * avg_weights

    test_policy = SimpleMLPPolicy(input_dim, output_dim)
    reward = evaluate(TreasureMazeEnv(), avg_weights, test_policy)
    log_episode(log_file, rnd, reward)
    episode_rewards.append(reward)
    print(f" Eval reward: {reward:.2f}")

save_reward_plot(episode_rewards, os.path.join(log_dir, "reward_plot.png"))
print(f"\n Done! Logs and plot saved to: {log_dir}")
