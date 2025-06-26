# federated_reinforcement_learning.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import copy

### --- NEW: Imports for logging and argument parsing --- ###
import argparse
import os
import csv

# --- Hyperparameters ---
NUM_WORKERS = 5
FEDERATED_ROUNDS = 50
LOCAL_EPISODES_PER_ROUND = 10 
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01

### --- NEW: Setup for Logging --- ###
LOG_DIR = "logs/federated_rl"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

# Setup CSV logging
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    # The metric here is the performance of the global model after each round
    writer.writerow(['round', 'average_global_reward'])

# --- Server (Central Aggregator) ---
global_q_table = np.zeros((GridEnvironment().grid_size, GridEnvironment().grid_size, 4))

# --- Worker Simulation ---
class Worker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.env = GridEnvironment()
        self.local_q_table = np.copy(global_q_table)

    def train_locally(self, epsilon):
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

### --- NEW: Function to evaluate the global model's performance --- ###
def evaluate_global_model(q_table):
    env = GridEnvironment()
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
workers = [Worker(i) for i in range(NUM_WORKERS)]

for round_num in range(FEDERATED_ROUNDS):
    local_models = []
    
    for worker in workers:
        worker.local_q_table = np.copy(global_q_table)

    # In a real scenario, this loop would be parallel
    for worker in workers:
        print(f"Round {round_num+1}/{FEDERATED_ROUNDS} - Worker {worker.id} training...")
        local_model = worker.train_locally(EPSILON)
        local_models.append(local_model)

    global_q_table = np.mean(local_models, axis=0)

    ### --- NEW: Evaluate and Log Performance --- ###
    global_reward = evaluate_global_model(global_q_table)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, global_reward])

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
    
    print(f"Round {round_num+1} complete. Global Reward: {global_reward:.2f}. Epsilon: {EPSILON:.4f}\n")

print("\n--- Federated Training Complete ---")


### --- MODIFIED: Final Demonstration is now conditional --- ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render final policy")
    args, unknown = parser.parse_known_args() # Use parse_known_args in case other args are passed

    if args.render:
        print("\n--- Demonstration of Final Global Model ---")
        env = GridEnvironment()
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