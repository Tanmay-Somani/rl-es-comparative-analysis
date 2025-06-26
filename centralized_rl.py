# centralised_reinforcement_learning.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment

### --- NEW: Imports for logging and argument parsing --- ###
import argparse
import os
import csv

# --- Hyperparameters ---
EPISODES = 2000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01

### --- NEW: Setup for Logging --- ###
LOG_DIR = "logs/centralized_rl"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

# Setup CSV logging - This will overwrite the file at the start of each run
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'total_reward'])

# --- Initialization ---
env = GridEnvironment()
# Q-table: states (6*6) x actions (4)
q_table = np.zeros((env.grid_size, env.grid_size, env.action_space_size))

# --- Training Loop ---
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0 ### <-- NEW: Initialize total reward for the episode

    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < EPSILON:
            action = random.randint(0, env.action_space_size - 1) # Explore
        else:
            action = np.argmax(q_table[state]) # Exploit

        new_state, reward, done = env.step(action)
        total_reward += reward ### <-- NEW: Accumulate reward

        # Q-learning formula
        old_value = q_table[state][action]
        next_max = np.max(q_table[new_state])
        
        new_q_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        q_table[state][action] = new_q_value
        
        state = new_state

    ### --- NEW: Log total reward after each episode --- ###
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward])

    # Decay epsilon
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    if episode % 200 == 0:
        print(f"Episode {episode}, Epsilon: {EPSILON:.4f}, Total Reward: {total_reward}")

print("\n--- Training Complete ---")

### --- MODIFIED: Final Demonstration is now conditional --- ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render final policy")
    # Use parse_known_args to avoid errors if other args are passed by a master script
    args, unknown = parser.parse_known_args()

    if args.render:
        print("\n--- Demonstration of Learned Policy ---")
        state = env.reset()
        done = False
        env.render("Centralized RL - Final Path")
        path_length = 0
        while not done and path_length < 25: # Safety break
            action = np.argmax(q_table[state])
            state, _, done = env.step(action)
            env.render("Centralized RL - Final Path")
            path_length += 1
            print(f"Action: {action}, Position: {state}")
            time.sleep(0.2)

        print("Demonstration finished.")