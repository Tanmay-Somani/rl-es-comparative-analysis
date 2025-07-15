# fl_rl.py
import gymnasium as gym
import ale_py
import shimmy
import numpy as np
import psutil
import time
from tqdm import tqdm

class RLAgent:
    """A client in the federated learning setup for Assault."""
    def __init__(self, env, n_bins=8):
        self.n_bins = n_bins
        self.bins = np.linspace(0, 255, n_bins+1)
        self.q_table = np.zeros((n_bins,) * 4 + (env.action_space.n,))  # Use first 4 RAM bytes
        self.learning_rate = 0.1
        self.discount_factor = 0.99

    def train(self, env, local_epochs):
        for _ in range(local_epochs):
            state, _ = env.reset()
            idxs = [np.digitize(state[i], self.bins)-1 for i in range(4)]
            idxs = [max(0, min(idx, self.n_bins-1)) for idx in idxs]
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = np.argmax(self.q_table[tuple(idxs)])
                new_state, reward, terminated, truncated, _ = env.step(action)
                new_idxs = [np.digitize(new_state[i], self.bins)-1 for i in range(4)]
                new_idxs = [max(0, min(idx, self.n_bins-1)) for idx in new_idxs]
                self.q_table[tuple(idxs)+(action,)] = self.q_table[tuple(idxs)+(action,)] * (1 - self.learning_rate) + \
                    (reward + self.discount_factor * np.max(self.q_table[tuple(new_idxs)])) * self.learning_rate
                idxs = new_idxs
        return self.q_table

def run_fl_rl(log_data, n_clients=5, n_rounds=100, local_epochs=3):
    print("\nRunning Federated RL (Q-learning) on Assault-v4 (RAM)...")
    start_time = time.time()
    process = psutil.Process()
    env = gym.make('Assault-v4', obs_type='ram')
    clients = [RLAgent(env) for _ in range(n_clients)]
    global_model = np.zeros_like(clients[0].q_table)
    communication_overhead = 0
    for _ in tqdm(range(n_rounds), desc="RL FL Assault"):
        local_models = []
        for client in clients:
            client.q_table = global_model.copy()
            local_model = client.train(env, local_epochs)
            local_models.append(local_model)
            communication_overhead += global_model.nbytes
        global_model = np.mean(local_models, axis=0)
        # Evaluate global model
        rewards = 0
        for _ in range(5):
            state, _ = env.reset()
            idxs = [np.digitize(state[i], clients[0].bins)-1 for i in range(4)]
            idxs = [max(0, min(idx, clients[0].n_bins-1)) for idx in idxs]
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = np.argmax(global_model[tuple(idxs)])
                state, reward, terminated, truncated, _ = env.step(action)
                idxs = [np.digitize(state[i], clients[0].bins)-1 for i in range(4)]
                idxs = [max(0, min(idx, clients[0].n_bins-1)) for idx in idxs]
                rewards += reward
        avg_reward = rewards / 5.0
        log_data['rl_fl']['cpu_usage'].append(process.cpu_percent())
        log_data['rl_fl']['memory_usage'].append(process.memory_info().rss / (1024 * 1024))
        log_data['rl_fl']['training_time'].append(time.time() - start_time)
        log_data['rl_fl']['convergence_speed'].append(avg_reward)
        log_data['rl_fl']['communication_overhead'].append(communication_overhead)
    env.close()
    return log_data 