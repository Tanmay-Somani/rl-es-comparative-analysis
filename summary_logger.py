import csv
import os
import numpy as np

def summarize_and_save(log_file, method_name, summary_file="logs/summary/summary.csv"):
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} not found.")
        return

    rewards = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            try:
                _, reward = line.strip().split(',')
                rewards.append(float(reward))
            except:
                continue

    if not rewards:
        print(f"No rewards found in {log_file}. Skipping summary.")
        return

    avg_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    summary_row = [method_name, round(avg_reward, 2), round(max_reward, 2), round(min_reward, 2)]

    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    write_header = not os.path.exists(summary_file)

    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Method", "Average Reward", "Max Reward", "Min Reward"])
        writer.writerow(summary_row)

    print(f"✔️ Summary saved for {method_name} in {summary_file}")
