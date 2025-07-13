# utils.py
import matplotlib.pyplot as plt

def plot_results(log_data):
    """
    Plots the benchmark results for all specified experiments.
    """
    print("\nPlotting results...")
    # Define colors for consistent plotting
    colors = {'rl_non_fl': 'blue', 'es_distributed': 'green', 'rl_fl': 'red'}
    labels = {'rl_non_fl': 'RL Non-FL', 'es_distributed': 'ES Distributed', 'rl_fl': 'RL FL'}

    # Plot CPU Usage
    plt.figure(figsize=(12, 6))
    for key in log_data:
        plt.plot(log_data[key]['cpu_usage'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes / Generations / Rounds')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Benchmark')
    plt.legend()
    plt.grid(True)
    plt.savefig('cpu_usage_benchmark.png')

    # Plot Memory Usage
    plt.figure(figsize=(12, 6))
    for key in log_data:
        plt.plot(log_data[key]['memory_usage'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes / Generations / Rounds')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Benchmark')
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_usage_benchmark.png')

    # Plot Training Time
    plt.figure(figsize=(12, 6))
    for key in log_data:
        plt.plot(log_data[key]['training_time'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes / Generations / Rounds')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Benchmark')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_time_benchmark.png')

    # Plot Convergence Speed
    plt.figure(figsize=(12, 6))
    for key in log_data:
        plt.plot(log_data[key]['convergence_speed'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes / Generations / Rounds')
    plt.ylabel('Average Reward')
    plt.title('Convergence Speed Benchmark')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_speed_benchmark.png')

    # Plot Communication Overhead (only for FL)
    plt.figure(figsize=(12, 6))
    plt.plot(log_data['rl_fl']['communication_overhead'], label=labels['rl_fl'], color=colors['rl_fl'])
    plt.xlabel('Rounds')
    plt.ylabel('Total Communication Overhead (bytes)')
    plt.title('Communication Overhead Benchmark (Federated RL)')
    plt.legend()
    plt.grid(True)
    plt.savefig('communication_overhead_benchmark.png')

    plt.show()
    print("Benchmarking complete. Plots saved to the current directory.")