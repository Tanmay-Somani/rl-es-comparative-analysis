# fl_rl.py
import gymnasium as gym
import numpy as np
import psutil
import time
from tqdm import tqdm

class RLAgent:
    """A client in the federated learning setup for CartPole."""
    def __init__(self, env):
        # Discretize the state space for the Q-table
        self.pos_bins = np.linspace(-4.8, 4.8, 10)  # Cart position
        self.vel_bins = np.linspace(-10, 10, 10)    # Cart velocity  
        self.angle_bins = np.linspace(-0.418, 0.418, 10)  # Pole angle (in radians)
        self.ang_vel_bins = np.linspace(-10, 10, 10)      # Pole angular velocity
        
        self.q_table = np.zeros((len(self.pos_bins), len(self.vel_bins), 
                                len(self.angle_bins), len(self.ang_vel_bins), env.action_space.n))
        self.learning_rate = 0.1
        self.discount_factor = 0.95

    def train(self, env, local_epochs):
        """Simulates local training on a client device."""
        for _ in range(local_epochs):
            state, _ = env.reset()
            
            # Discretize state
            state_p = np.digitize(state[0], self.pos_bins) - 1
            state_v = np.digitize(state[1], self.vel_bins) - 1
            state_a = np.digitize(state[2], self.angle_bins) - 1
            state_av = np.digitize(state[3], self.ang_vel_bins) - 1
            
            # Clamp to valid indices
            state_p = max(0, min(state_p, len(self.pos_bins) - 1))
            state_v = max(0, min(state_v, len(self.vel_bins) - 1))
            state_a = max(0, min(state_a, len(self.angle_bins) - 1))
            state_av = max(0, min(state_av, len(self.ang_vel_bins) - 1))
            
            terminated, truncated = False, False
            while not (terminated or truncated):
                # For local training, we assume the client uses its current policy
                action = np.argmax(self.q_table[state_p, state_v, state_a, state_av, :])
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                # Discretize new state
                new_state_p = np.digitize(new_state[0], self.pos_bins) - 1
                new_state_v = np.digitize(new_state[1], self.vel_bins) - 1
                new_state_a = np.digitize(new_state[2], self.angle_bins) - 1
                new_state_av = np.digitize(new_state[3], self.ang_vel_bins) - 1
                
                # Clamp to valid indices
                new_state_p = max(0, min(new_state_p, len(self.pos_bins) - 1))
                new_state_v = max(0, min(new_state_v, len(self.vel_bins) - 1))
                new_state_a = max(0, min(new_state_a, len(self.angle_bins) - 1))
                new_state_av = max(0, min(new_state_av, len(self.ang_vel_bins) - 1))
                
                # Q-table update
                self.q_table[state_p, state_v, state_a, state_av, action] = \
                    self.q_table[state_p, state_v, state_a, state_av, action] * (1 - self.learning_rate) + \
                    (reward + self.discount_factor * np.max(self.q_table[new_state_p, new_state_v, new_state_a, new_state_av, :])) * self.learning_rate
                
                state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
        return self.q_table

def run_fl_rl(log_data, n_clients=5, n_rounds=500, local_epochs=5):
    """Runs the main federated Q-learning simulation for CartPole."""
    print("\nRunning Federated Reinforcement Learning on CartPole...")
    start_time = time.time()
    process = psutil.Process()
    env = gym.make('CartPole-v1')

    clients = [RLAgent(env) for _ in range(n_clients)]
    global_model = np.zeros_like(clients[0].q_table)
    communication_overhead = 0

    for _ in tqdm(range(n_rounds), desc="RL FL CartPole"):
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
                # Discretize state
                state_p = np.digitize(state[0], clients[0].pos_bins) - 1
                state_v = np.digitize(state[1], clients[0].vel_bins) - 1
                state_a = np.digitize(state[2], clients[0].angle_bins) - 1
                state_av = np.digitize(state[3], clients[0].ang_vel_bins) - 1
                
                # Clamp to valid indices
                state_p = max(0, min(state_p, len(clients[0].pos_bins) - 1))
                state_v = max(0, min(state_v, len(clients[0].vel_bins) - 1))
                state_a = max(0, min(state_a, len(clients[0].angle_bins) - 1))
                state_av = max(0, min(state_av, len(clients[0].ang_vel_bins) - 1))
                
                action = np.argmax(global_model[state_p, state_v, state_a, state_av, :])
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