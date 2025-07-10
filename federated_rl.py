# federated_reinforcement_learning.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import copy
import argparse
import os
import csv
import pickle

# --- Hyperparameters ---
NUM_WORKERS = 5
FEDERATED_ROUNDS = 300
LOCAL_EPISODES_PER_ROUND = 10 
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.90
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# --- Argument Parsing for Dynamic Logging ---
parser = argparse.ArgumentParser()
parser.add_argument("--log_file", type=str, help="Path to save the training log CSV.")
parser.add_argument("--render", action="store_true", help="Render final policy.")
args, unknown = parser.parse_known_args()

# --- Setup for Logging ---
if args.log_file:
    LOG_FILE = args.log_file
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
else:
    LOG_DIR = "logs/federated_rl"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round', 'average_global_reward', 'elapsed_time'])

# --- Worker Simulation ---
class Worker:
    def __init__(self, worker_id, env_config):
        self.id = worker_id
        self.env = GridEnvironment()
        self.env.set_config(env_config)
        self.local_q_table = None

    def train_locally(self, epsilon, global_q_table):
        self.local_q_table = np.copy(global_q_table)
        for _ in range(LOCAL_EPISODES_PER_ROUND):
            state = self.env.reset()
            done = False
            steps = 0
            while not done and steps < 100:
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, self.env.action_space_size - 1)
                else:
                    action = np.argmax(self.local_q_table[state])
                new_state, reward, done = self.env.step(action)
                steps += 1
                old_value = self.local_q_table[state][action]
                next_max = np.max(self.local_q_table[new_state])
                new_q_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
                self.local_q_table[state][action] = new_q_value
                state = new_state
        return self.local_q_table

def evaluate_global_model(q_table, env_config):
    env = GridEnvironment()
    env.set_config(env_config)
    total_reward = 0
    for _ in range(5): # Average over 5 episodes for stability
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        while not done and steps < 100:
            action = np.argmax(q_table[state])
            state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1
        total_reward += episode_reward
    return total_reward / 5

# --- Federated Training Loop ---
master_env = GridEnvironment()
master_env_config = master_env.get_config()
global_q_table = np.zeros((master_env.grid_size, master_env.grid_size, master_env.action_space_size))
workers = [Worker(i, master_env_config) for i in range(NUM_WORKERS)]
start_time = time.time()

for round_num in range(FEDERATED_ROUNDS):
    local_models = [worker.train_locally(EPSILON, global_q_table) for worker in workers]
    global_q_table = np.mean(local_models, axis=0)
    
    global_reward = evaluate_global_model(global_q_table, master_env_config)
    elapsed_time = time.time() - start_time
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, global_reward, elapsed_time])

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
    
    if round_num % 50 == 0:
        print(f"Round {round_num+1}/{FEDERATED_ROUNDS} | Global Reward: {global_reward:.2f} | Epsilon: {EPSILON:.4f}")

print("\n--- Federated Training Complete ---")

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
model_data = {'model': global_q_table, 'env_config': master_env_config}
model_path = os.path.join(MODEL_DIR, "federated_rl.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Global Q-table saved to {model_path}")

if args.render:
    print("\n--- Demonstration of Final Global Model ---")
    env = GridEnvironment()
    env.set_config(master_env_config)
    state = env.reset()
    done = False
    env.render("Federated RL - Final Path")
    for _ in range(30):
        if done: break
        action = np.argmax(global_q_table[state])
        state, _, done = env.step(action)
        env.render("Federated RL - Final Path")
        time.sleep(0.2)
    print("Demonstration finished.")