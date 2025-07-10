# federated_reinforcement_learning.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import copy
import argparse
import os
import csv
import pickle ### NEW: Import pickle for saving the model
import time 
# --- Hyperparameters ---
NUM_WORKERS = 5
FEDERATED_ROUNDS = 100
LOCAL_EPISODES_PER_ROUND = 10 
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.80
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01

# --- Setup for Logging ---
LOG_DIR = "logs/federated_rl"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round', 'average_global_reward','elapsed_time'])

# --- Worker Simulation ---
### MODIFIED: Worker now accepts an environment configuration ###
class Worker:
    def __init__(self, worker_id, env_config):
        self.id = worker_id
        self.env = GridEnvironment()
        self.env.set_config(env_config) # Set to master config
        self.local_q_table = None # Will be set by server each round

    def train_locally(self, epsilon, global_q_table):
        # Receive the latest global model
        self.local_q_table = np.copy(global_q_table)

        for _ in range(LOCAL_EPISODES_PER_ROUND):
            state = self.env.reset()
            done = False
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, self.env.action_space_size - 1)
                else:
                    action = np.argmax(self.local_q_table[state])
                
                new_state, reward, done = self.env.step(action)
                
                old_value = self.local_q_table[state][action]
                next_max = np.max(self.local_q_table[new_state])
                new_q_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
                self.local_q_table[state][action] = new_q_value
                state = new_state
        return self.local_q_table

### MODIFIED: Function to evaluate the global model's performance on the correct grid ###
def evaluate_global_model(q_table, env_config):
    env = GridEnvironment()
    env.set_config(env_config) # Use the specific config for evaluation
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < 50:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward

# --- Federated Training Loop ---
### NEW: Create a single master environment for the run ###
master_env = GridEnvironment()
master_env_config = master_env.get_config()
start_time = time.time()
### MODIFIED: Server holds the global model ###
global_q_table = np.zeros((master_env.grid_size, master_env.grid_size, master_env.action_space_size))

### MODIFIED: Initialize workers with the same environment config ###
workers = [Worker(i, master_env_config) for i in range(NUM_WORKERS)]

for round_num in range(FEDERATED_ROUNDS):
    local_models = []
    
    # In a real scenario, this loop would be parallel
    for worker in workers:
        # Pass the global model to the worker for local training
        local_model = worker.train_locally(EPSILON, global_q_table)
        local_models.append(local_model)

    # Aggregate local models (Federated Averaging)
    global_q_table = np.mean(local_models, axis=0)

    # Evaluate and Log Performance
    elapsed_time = time.time() - start_time
    global_reward = evaluate_global_model(global_q_table, master_env_config)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, global_reward,elapsed_time])

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
    
    if round_num % 20 == 0:
        print(f"Round {round_num+1} complete. Global Reward: {global_reward:.2f}. Epsilon: {EPSILON:.4f}")

print("\n--- Federated Training Complete ---")

### NEW: Save the final global Q-table and its environment ###
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
model_data = {
    'model': global_q_table,
    'env_config': master_env_config
}
model_path = os.path.join(MODEL_DIR, "federated_rl.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Global Q-table saved to {model_path}")


### MODIFIED: Final Demonstration uses the master environment ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render final policy")
    args, unknown = parser.parse_known_args()

    if args.render:
        print("\n--- Demonstration of Final Global Model ---")
        # Use the same environment configuration that the model was trained on
        env = GridEnvironment()
        env.set_config(master_env_config)
        state = env.reset()
        done = False
        env.render("Federated RL - Final Path")
        path_length = 0
        while not done and path_length < 25:
            action = np.argmax(global_q_table[state])
            state, _, done = env.step(action)
            env.render("Federated RL - Final Path")
            path_length += 1
            print(f"Action: {action}, Position: {state}")
            time.sleep(0.2)
        print("Demonstration finished.")