import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import csv
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Suppress a known warning from the gym library
warnings.filterwarnings("ignore", category=DeprecationWarning, module='gym.utils.passive_env_checker')

# --- Config ---
ENV_NAME = "PongNoFrameskip-v4"
POPULATION_SIZE = 32
SIGMA = 0.1
ALPHA = 0.02
SEED = 42
LOG_DIR = "es_breakout_logs" # Renamed for clarity since we are using Breakout
CHECKPOINT_PATH = os.path.join(LOG_DIR, "es_checkpoint.npy")
REWARD_LOG = os.path.join(LOG_DIR, "es_rewards.csv")
MODEL_PATH = os.path.join(LOG_DIR, "es_atari_policy.pth")
PLOT_PATH = os.path.join(LOG_DIR, "es_rewards_plot.png")
START_GEN = 0
MAX_ITER = 10 # Set to a low number for quick testing

# --- Create log directory ---
os.makedirs(LOG_DIR, exist_ok=True)

# --- Set seeds ---
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# --- Preprocessing wrapper ---
def preprocess(obs):
    """
    Preprocesses a 210x160x3 frame from Atari into a 1D 6400-element vector.
    """
    obs = obs[35:195]  # Crop the score bar and border
    obs = obs[::2, ::2, 0]  # Downsample and take one color channel
    obs[obs == 144] = 0  # Erase background
    obs[obs == 109] = 0  # Erase background
    obs[obs != 0] = 1  # Set paddles and ball to 1
    return obs.astype(np.float32).ravel()

# --- Policy Network ---
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- Evaluate one rollout ---
def evaluate(env, model, render=False):
    obs, _ = env.reset()
    obs = preprocess(obs)
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(obs_tensor)
        action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = preprocess(obs)
        total_reward += reward
        if render:
            env.render()
    return total_reward

# --- Save logs ---
def log_reward(generation, mean_reward):
    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(REWARD_LOG), exist_ok=True)
    write_header = not os.path.exists(REWARD_LOG)
    with open(REWARD_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["generation", "mean_reward"])
        writer.writerow([generation, mean_reward])

# --- Plot rewards ---
def plot_rewards():
    """
    Reads the reward log and plots the mean reward per generation.
    """
    if not os.path.exists(REWARD_LOG):
        # This check is important, especially for the very first iteration
        logging.warning("Reward log file not found. Skipping plotting for now.")
        return

    # To avoid errors with an empty file on the first run
    if os.path.getsize(REWARD_LOG) == 0:
        logging.info("Reward log is empty. Skipping plotting for now.")
        return

    logging.info(f"Updating plot from {REWARD_LOG}...")
    
    data = pd.read_csv(REWARD_LOG)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data["generation"], data["mean_reward"], marker='o', linestyle='-', label="Mean Reward per Generation")
    
    plt.title("Training Progress: Mean Reward vs. Generation")
    plt.xlabel("Generation")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.legend()
    
    plt.savefig(PLOT_PATH)
    plt.close() # Close the figure to free up memory

# --- Train with ES ---
def train_es(env_name, iterations=MAX_ITER):
    env = gym.make(env_name)

    initial_obs, _ = env.reset()
    obs = preprocess(initial_obs)
    input_dim = obs.shape[0]
    output_dim = env.action_space.n

    mu = PolicyNet(input_dim, output_dim).float()

    if os.path.exists(CHECKPOINT_PATH):
        logging.info(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        mu_params = np.load(CHECKPOINT_PATH).astype(np.float32)
        nn.utils.vector_to_parameters(torch.from_numpy(mu_params), mu.parameters())
    else:
        logging.info("Starting new training run.")
        mu_params = nn.utils.parameters_to_vector(mu.parameters()).detach().numpy()

    for gen in range(START_GEN, START_GEN + iterations):
        noise_list = []
        rewards = []

        for i in range(POPULATION_SIZE):
            noise = np.random.randn(*mu_params.shape).astype(np.float32)
            model = PolicyNet(input_dim, output_dim).float()
            noisy_params = mu_params + SIGMA * noise
            nn.utils.vector_to_parameters(torch.from_numpy(noisy_params), model.parameters())
            
            reward = evaluate(env, model)
            noise_list.append(noise)
            rewards.append(reward)

        rewards = np.array(rewards)
        if np.std(rewards) > 1e-8:
            normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        else:
            normalized_rewards = rewards - np.mean(rewards)
            
        gradient = np.dot(np.array(noise_list).T, normalized_rewards) / POPULATION_SIZE
        mu_params += ALPHA * gradient

        mean_reward = np.mean(rewards)
        logging.info(f"Generation {gen} | Mean Reward: {mean_reward:.2f}")
        
        # --- MODIFICATION ---
        # Log the reward and then immediately update the plot
        log_reward(gen, mean_reward)
        plot_rewards() 
        # --- END MODIFICATION ---
        
        np.save(CHECKPOINT_PATH, mu_params)

    logging.info(f"Training complete. Saving final model to {MODEL_PATH}")
    final_model = PolicyNet(input_dim, output_dim).float()
    nn.utils.vector_to_parameters(torch.from_numpy(mu_params), final_model.parameters())
    torch.save(final_model.state_dict(), MODEL_PATH)
    env.close()

if __name__ == "__main__":
    # Run the training process. The plotting is now handled inside train_es.
    train_es(ENV_NAME)
    
    logging.info("Script finished. Final plot is available at " + PLOT_PATH)