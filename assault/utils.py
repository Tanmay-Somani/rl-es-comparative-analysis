# utils.py
import matplotlib.pyplot as plt

def plot_results(log_data):
    """
    Plots the benchmark results for all specified experiments on Assault.
    """
    print("\nPlotting Assault results...")
    colors = {
        'rl_non_fl': 'blue',
        'es_distributed': 'green',
        'rl_fl': 'red',
        'es_fl': 'purple'
    }
    labels = {
        'rl_non_fl': 'RL Non-FL',
        'es_distributed': 'ES Distributed',
        'rl_fl': 'RL FL',
        'es_fl': 'ES FL'
    }

    # Plot CPU Usage
    plt.figure(figsize=(12, 6))
    for key in log_data:
        if key in colors:
            plt.plot(log_data[key]['cpu_usage'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes / Generations / Rounds')
    plt.ylabel('CPU Usage (%)')
    plt.title('Assault - CPU Usage Benchmark')
    plt.legend()
    plt.grid(True)
    plt.savefig('assault_cpu_usage_benchmark.png')

    # Plot Memory Usage
    plt.figure(figsize=(12, 6))
    for key in log_data:
        if key in colors:
            plt.plot(log_data[key]['memory_usage'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes / Generations / Rounds')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Assault - Memory Usage Benchmark')
    plt.legend()
    plt.grid(True)
    plt.savefig('assault_memory_usage_benchmark.png')

    # Plot Training Time
    plt.figure(figsize=(12, 6))
    for key in log_data:
        if key in colors:
            plt.plot(log_data[key]['training_time'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes / Generations / Rounds')
    plt.ylabel('Training Time (s)')
    plt.title('Assault - Training Time Benchmark')
    plt.legend()
    plt.grid(True)
    plt.savefig('assault_training_time_benchmark.png')

    # Plot Convergence Speed
    plt.figure(figsize=(12, 6))
    for key in log_data:
        if key in colors:
            plt.plot(log_data[key]['convergence_speed'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes / Generations / Rounds')
    plt.ylabel('Average Reward')
    plt.title('Assault - Convergence Speed Benchmark')
    plt.legend()
    plt.grid(True)
    plt.savefig('assault_convergence_speed_benchmark.png')

    # Plot Communication Overhead (only for FL methods)
    plt.figure(figsize=(12, 6))
    for key in ['rl_fl', 'es_fl']:
        if key in log_data and 'communication_overhead' in log_data[key]:
            plt.plot(log_data[key]['communication_overhead'], label=labels[key], color=colors[key])
    plt.xlabel('Rounds')
    plt.ylabel('Total Communication Overhead (bytes)')
    plt.title('Assault - Communication Overhead Benchmark (Federated Methods)')
    plt.legend()
    plt.grid(True)
    plt.savefig('assault_communication_overhead_benchmark.png')

    # Combined comparison plot
    plt.figure(figsize=(15, 10))
    # Subplot 1: Convergence Speed
    plt.subplot(2, 3, 1)
    for key in log_data:
        if key in colors:
            plt.plot(log_data[key]['convergence_speed'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes/Generations/Rounds')
    plt.ylabel('Average Reward')
    plt.title('Convergence Speed')
    plt.legend()
    plt.grid(True)
    # Subplot 2: CPU Usage
    plt.subplot(2, 3, 2)
    for key in log_data:
        if key in colors:
            plt.plot(log_data[key]['cpu_usage'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes/Generations/Rounds')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage')
    plt.legend()
    plt.grid(True)
    # Subplot 3: Memory Usage
    plt.subplot(2, 3, 3)
    for key in log_data:
        if key in colors:
            plt.plot(log_data[key]['memory_usage'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes/Generations/Rounds')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage')
    plt.legend()
    plt.grid(True)
    # Subplot 4: Training Time
    plt.subplot(2, 3, 4)
    for key in log_data:
        if key in colors:
            plt.plot(log_data[key]['training_time'], label=labels[key], color=colors[key])
    plt.xlabel('Episodes/Generations/Rounds')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time')
    plt.legend()
    plt.grid(True)
    # Subplot 5: Final Performance Comparison
    plt.subplot(2, 3, 5)
    final_rewards = []
    method_names = []
    for key in log_data:
        if key in colors and log_data[key]['convergence_speed']:
            final_rewards.append(log_data[key]['convergence_speed'][-1])
            method_names.append(labels[key])
    bars = plt.bar(method_names, final_rewards, color=[colors.get(key, 'gray') for key in log_data.keys()])
    plt.ylabel('Final Average Reward')
    plt.title('Final Performance Comparison')
    plt.xticks(rotation=45)
    for bar, reward in zip(bars, final_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{reward:.1f}', ha='center', va='bottom')
    # Subplot 6: Communication Overhead (if available)
    plt.subplot(2, 3, 6)
    fl_methods = [key for key in ['rl_fl', 'es_fl'] if key in log_data and 'communication_overhead' in log_data[key]]
    if fl_methods:
        for key in fl_methods:
            plt.plot(log_data[key]['communication_overhead'], label=labels[key], color=colors[key])
        plt.xlabel('Rounds')
        plt.ylabel('Communication Overhead (bytes)')
        plt.title('Communication Overhead')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No FL data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Communication Overhead')
    plt.tight_layout()
    plt.savefig('assault_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Assault benchmarking complete. Plots saved to the current directory.") 