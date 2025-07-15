# rl_non_fl.py
import gymnasium as gym
import ale_py
import shimmy
import numpy as np
import psutil
import time
from tqdm import tqdm

def run_rl_non_fl(log_data):
    """
    Runs a non-federated Q-learning agent on the Assault-v4 environment using RAM observations.
    """
    print("Running Non-Federated RL (Q-learning) on Assault-v4 (RAM)...")
    start_time = time.time()
    process = psutil.Process()
    env = gym.make('Assault-v4', obs_type='ram')

    # Discretize the RAM state space for the Q-table
    # RAM is 128 bytes, each in [0,255]. We'll use 8 bins per byte for tractability.
    n_bins = 8
    bins = np.linspace(0, 255, n_bins+1)
    q_table = np.zeros((n_bins**4, env.action_space.n))  # Use only first 4 RAM bytes for Q-table

    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay_rate = 0.995
    rng = np.random.default_rng()
    n_episodes = 500

    for episode in tqdm(range(n_episodes), desc="RL Non-FL Assault"):
        state, _ = env.reset()
        # Discretize first 4 RAM bytes
        discrete_state = [np.digitize(state[i], bins)-1 for i in range(4)]
        discrete_state = [max(0, min(idx, n_bins-1)) for idx in discrete_state]
        state_index = discrete_state[0] * n_bins**3 + discrete_state[1] * n_bins**2 + discrete_state[2] * n_bins + discrete_state[3]
        terminated = False
        truncated = False
        rewards = 0

        while not (terminated or truncated):
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_index])

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_discrete_state = [np.digitize(new_state[i], bins)-1 for i in range(4)]
            new_discrete_state = [max(0, min(idx, n_bins-1)) for idx in new_discrete_state]
            new_state_index = new_discrete_state[0] * n_bins**3 + new_discrete_state[1] * n_bins**2 + new_discrete_state[2] * n_bins + new_discrete_state[3]

            # Q-table update rule
            q_table[state_index, action] = q_table[state_index, action] + learning_rate * \
                (reward + discount_factor * np.max(q_table[new_state_index]) - q_table[state_index, action])
            state_index = new_state_index
            rewards += reward

        epsilon = max(epsilon * epsilon_decay_rate, 0.05)
        # Log metrics for this episode
        log_data['rl_non_fl']['cpu_usage'].append(process.cpu_percent())
        log_data['rl_non_fl']['memory_usage'].append(process.memory_info().rss / (1024 * 1024))
        log_data['rl_non_fl']['training_time'].append(time.time() - start_time)
        log_data['rl_non_fl']['convergence_speed'].append(rewards)

    env.close()
    return log_data 