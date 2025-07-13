# fl_rl.py
import gymnasium as gym
import numpy as np
import psutil
import time
from tqdm import tqdm

class RLAgent:
    """A client in the federated learning setup."""
    def __init__(self, env):
        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
        self.q_table = np.zeros((len(self.pos_space), len(self.vel_space), env.action_space.n))
        self.learning_rate = 0.1
        self.discount_factor = 0.95

    def train(self, env, local_epochs):
        """Simulates local training on a client device."""
        for _ in range(local_epochs):
            state, _ = env.reset()
            state_p = np.digitize(state[0], self.pos_space)
            state_v = np.digitize(state[1], self.vel_space)
            terminated, truncated = False, False
            while not (terminated or truncated):
                # For local training, we assume the client uses its current policy
                action = np.argmax(self.q_table[state_p, state_v, :])
                new_state, reward, terminated, truncated, _ = env.step(action)
                new_state_p = np.digitize(new_state[0], self.pos_space)
                new_state_v = np.digitize(new_state[1], self.vel_space)
                # Q-table update
                self.q_table[state_p, state_v, action] = self.q_table[state_p, state_v, action] * (1 - self.learning_rate) + \
                    (reward + self.discount_factor * np.max(self.q_table[new_state_p, new_state_v, :])) * self.learning_rate
                state_p, state_v = new_state_p, new_state_v
        return self.q_table

def run_fl_rl(log_data, n_clients=5, n_rounds=2000, local_epochs=5):
    """Runs the main federated Q-learning simulation."""
    print("\nRunning Federated Reinforcement Learning...")
    start_time = time.time()
    process = psutil.Process()
    env = gym.make('MountainCar-v0')

    clients = [RLAgent(env) for _ in range(n_clients)]
    global_model = np.zeros_like(clients[0].q_table)
    communication_overhead = 0

    for _ in tqdm(range(n_rounds), desc="RL FL"):
        local_models = []
        for client in clients:
            # Send global model to client
            client.q_table = global_model.copy()
            # Client performs local training
            local_model = client.train(env, local_epochs)
            local_models.append(local_model)
            # Simulate communication cost (upload)
            communication_overhead += global_model.nbytes

        # Aggregate models using Federated Averaging
        global_model = np.mean(local_models, axis=0)

        # Evaluate the new global model
        rewards = 0
        for _ in range(10):  # 10 evaluation episodes
            state, _ = env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated):
                state_p = np.digitize(state[0], clients[0].pos_space)
                state_v = np.digitize(state[1], clients[0].vel_space)
                action = np.argmax(global_model[state_p, state_v, :])
                state, reward, terminated, truncated, _ = env.step(action)
                rewards += reward
        avg_reward = rewards / 10.0

        # Log metrics for this round
        log_data['rl_fl']['cpu_usage'].append(process.cpu_percent())
        log_data['rl_fl']['memory_usage'].append(process.memory_info().rss / (1024 * 1024))
        log_data['rl_fl']['training_time'].append(time.time() - start_time)
        log_data['rl_fl']['convergence_speed'].append(avg_reward)
        log_data['rl_fl']['communication_overhead'].append(communication_overhead)

    env.close()
    return log_data