import os
import matplotlib.pyplot as plt
import numpy as np

def save_reward_plot(rewards, filename="logs/plot.png", label="Reward"):
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def log_episode(log_file, episode, reward):
    with open(log_file, 'a') as f:
        f.write(f"{episode},{reward}\n")
