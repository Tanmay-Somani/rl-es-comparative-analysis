# centralised_reinforcement_learning.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import argparse
import os
import csv
import pickle ### NEW: Import pickle for saving the model

# --- Hyperparameters ---
EPISODES = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01

# --- Setup for Logging ---
LOG_DIR = "logs/centralized_rl"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'total_reward','elapsed_time'])

# --- Initialization ---
env = GridEnvironment() # Creates a single, randomized environment for this entire run
q_table = np.zeros((env.grid_size, env.grid_size, env.action_space_size))
start_time = time.time()

# --- Training Loop ---
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if random.uniform(0, 1) < EPSILON:
            action = random.randint(0, env.action_space_size - 1)
        else:
            action = np.argmax(q_table[state])

        new_state, reward, done = env.step(action)
        total_reward += reward

        old_value = q_table[state][action]
        next_max = np.max(q_table[new_state])
        
        new_q_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        q_table[state][action] = new_q_value
        
        state = new_state
    elapsed_time = time.time() - start_time
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward,elapsed_time])

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    if episode % 200 == 0:
        print(f"Episode {episode}, Epsilon: {EPSILON:.4f}, Total Reward: {total_reward}")

print("\n--- Training Complete ---")

### NEW: Save the final Q-table and the environment it was trained on ###
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
model_data = {
    'model': q_table,
    'env_config': env.get_config()
}
model_path = os.path.join(MODEL_DIR, "centralized_rl.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Q-table saved to {model_path}")


### MODIFIED: Final Demonstration is now conditional --- ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render final policy")
    args, unknown = parser.parse_known_args()

    if args.render:
        print("\n--- Demonstration of Learned Policy ---")
        state = env.reset()
        done = False
        env.render("Centralized RL - Final Path")
        path_length = 0
        while not done and path_length < 25:
            action = np.argmax(q_table[state])
            state, _, done = env.step(action)
            env.render("Centralized RL - Final Path")
            path_length += 1
            print(f"Action: {action}, Position: {state}")
            time.sleep(0.2)
        print("Demonstration finished.")