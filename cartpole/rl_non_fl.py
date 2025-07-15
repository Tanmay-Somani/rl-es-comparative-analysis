# rl_non_fl.py
import gymnasium as gym
import numpy as np
import psutil
import time
from tqdm import tqdm

def run_rl_non_fl(log_data):
    """
    Runs a non-federated Q-learning agent on the CartPole-v1 environment.
    """
    print("Running Non-Federated Reinforcement Learning (Q-learning) on CartPole...")
    start_time = time.time()
    process = psutil.Process()
    env = gym.make('CartPole-v1')

    # Discretize the state space for the Q-table
    # CartPole has 4 continuous state variables: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    pos_bins = np.linspace(-2.4, 2.4, 10)   # Cart position
    vel_bins = np.linspace(-5, 5, 10)    # Cart velocity  
    angle_bins = np.linspace(-0.2095, 0.2095, 10)  # Pole angle (in radians)
    ang_vel_bins = np.linspace(-5, 5, 10)       # Pole angular velocity
    
    q_table = np.zeros((len(pos_bins)+1, len(vel_bins)+1, len(angle_bins)+1, len(ang_vel_bins)+1, env.action_space.n))
    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay_rate = 0.995
    rng = np.random.default_rng()
    n_episodes = 1000

    for episode in tqdm(range(n_episodes), desc="RL Non-FL CartPole"):
        state, _ = env.reset()
        
        # Discretize state
        state_p = np.digitize(state[0], pos_bins)
        state_v = np.digitize(state[1], vel_bins) 
        state_a = np.digitize(state[2], angle_bins) 
        state_av = np.digitize(state[3], ang_vel_bins) 
        
        # Clamp to valid indices
        state_p = max(0, min(state_p, len(pos_bins) ))
        state_v = max(0, min(state_v, len(vel_bins) ))
        state_a = max(0, min(state_a, len(angle_bins) ))
        state_av = max(0, min(state_av, len(ang_vel_bins) ))
        
        terminated = False
        truncated = False
        rewards = 0

        while not (terminated or truncated):
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Discretize new state
            new_state_p = np.digitize(new_state[0], pos_bins) 
            new_state_v = np.digitize(new_state[1], vel_bins) 
            new_state_a = np.digitize(new_state[2], angle_bins) 
            new_state_av = np.digitize(new_state[3], ang_vel_bins) 
            
            # Clamp to valid indices
            new_state_p = max(0, min(new_state_p, len(pos_bins) ))
            new_state_v = max(0, min(new_state_v, len(vel_bins) ))
            new_state_a = max(0, min(new_state_a, len(angle_bins) ))
            new_state_av = max(0, min(new_state_av, len(ang_vel_bins)))

            # Q-table update rule
            q_table[state_p, state_v, state_a, state_av, action] = \
                q_table[state_p, state_v, state_a, state_av, action] * (1 - learning_rate) + \
                (reward + discount_factor * np.max(q_table[new_state_p, new_state_v, new_state_a, new_state_av, :])) * learning_rate

            state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
            rewards += reward

        epsilon = max(epsilon * epsilon_decay_rate, 0.01)
        
        # Log metrics for this episode
        log_data['rl_non_fl']['cpu_usage'].append(process.cpu_percent())
        log_data['rl_non_fl']['memory_usage'].append(process.memory_info().rss / (1024 * 1024))
        log_data['rl_non_fl']['training_time'].append(time.time() - start_time)
        log_data['rl_non_fl']['convergence_speed'].append(rewards)

    env.close()
    return log_data 