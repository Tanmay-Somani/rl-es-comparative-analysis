# centralised_reinforcement_learning.py
import numpy as np
import random
import time
from grid_headless import GridEnvironment
import argparse
import os
import csv
import pickle
from collections import deque

# --- Hyperparameters ---
EPISODES = 300 
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.8
EPSILON = 1.0
EPSILON_DECAY = 0.995 
MIN_EPSILON = 0.02

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
    LOG_DIR = "logs/centralized_rl"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'total_reward', 'elapsed_time'])

# --- Initialization ---
env = GridEnvironment() 
q_table = np.zeros((env.grid_size, env.grid_size, env.action_space_size))
latest_rewards = deque(maxlen=100)
solved_threshold = 85 
start_time = time.time()

# --- Training Loop ---
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done and steps < 100:
        if random.uniform(0, 1) < EPSILON:
            action = random.randint(0, env.action_space_size - 1)
        else:
            action = np.argmax(q_table[state])
        new_state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        old_value = q_table[state][action]
        next_max = np.max(q_table[new_state])
        new_q_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        q_table[state][action] = new_q_value
        state = new_state

    elapsed_time = time.time() - start_time
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward, elapsed_time])

    latest_rewards.append(total_reward)
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    if episode % 500 == 0:
        avg_reward = sum(latest_rewards) / len(latest_rewards)
        print(f"Episode {episode}/{EPISODES} | Epsilon: {EPSILON:.4f} | Avg Reward (last 100): {avg_reward:.2f}")

    if len(latest_rewards) == 100:
        average_reward_100 = sum(latest_rewards) / 100
        if average_reward_100 >= solved_threshold:
            print(f"\nEnvironment solved in {episode} episodes!")
            break

print("\n--- Training Complete ---")

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
model_data = {'model': q_table, 'env_config': env.get_config()}
model_path = os.path.join(MODEL_DIR, "centralized_rl.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Q-table saved to {model_path}")

if args.render:
    print("\n--- Demonstration of Learned Policy ---")
    state = env.reset()
    done = False
    env.render("Centralized RL - Final Path")
    for _ in range(30):
        if done: break
        action = np.argmax(q_table[state])
        state, _, done = env.step(action)
        env.render("Centralized RL - Final Path")
        time.sleep(0.2)
    print("Demonstration finished.")