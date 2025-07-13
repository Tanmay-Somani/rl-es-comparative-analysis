# rl_non_fl.py
import gymnasium as gym
import numpy as np
import psutil
import time
from tqdm import tqdm

def run_rl_non_fl(log_data):
    """
    Runs a non-federated Q-learning agent on the MountainCar-v0 environment.
    """
    print("Running Non-Federated Reinforcement Learning (Q-learning)...")
    start_time = time.time()
    process = psutil.Process()
    env = gym.make('MountainCar-v0')

    # Discretize the state space for the Q-table
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay_rate = 2 / 10000
    rng = np.random.default_rng()
    n_episodes = 10000

    for episode in tqdm(range(n_episodes), desc="RL Non-FL"):
        state, _ = env.reset()
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        
        terminated = False
        truncated = False
        rewards = 0

        while not (terminated or truncated):
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_p, state_v, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            # Q-table update rule
            q_table[state_p, state_v, action] = q_table[state_p, state_v, action] * (1 - learning_rate) + \
                (reward + discount_factor * np.max(q_table[new_state_p, new_state_v, :])) * learning_rate

            state_p, state_v = new_state_p, new_state_v
            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        # Log metrics for this episode
        log_data['rl_non_fl']['cpu_usage'].append(process.cpu_percent())
        log_data['rl_non_fl']['memory_usage'].append(process.memory_info().rss / (1024 * 1024))
        log_data['rl_non_fl']['training_time'].append(time.time() - start_time)
        log_data['rl_non_fl']['convergence_speed'].append(rewards)

    env.close()
    return log_data