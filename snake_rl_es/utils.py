# utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Game Configuration
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 480
BLOCK_SIZE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

# Game Speed
SPEED = 40

def plot_results(log_file, title):
    """
    Plots the benchmark results from a log file.
    """
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)

    # Plot Score
    sns.lineplot(ax=axes[0, 0], data=df, x=df.index, y='score', color='g')
    axes[0, 0].set_title('Game Score over Episodes/Generations')
    axes[0, 0].set_xlabel('Episode / Generation')
    axes[0, 0].set_ylabel('Score')

    # Plot CPU Usage
    sns.lineplot(ax=axes[0, 1], data=df, x='time_elapsed', y='cpu_usage', color='r')
    axes[0, 1].set_title('CPU Usage Over Time')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('CPU Usage (%)')

    # Plot Memory Usage
    sns.lineplot(ax=axes[1, 0], data=df, x='time_elapsed', y='memory_usage', color='b')
    axes[1, 0].set_title('Memory Usage Over Time')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Memory Usage (%)')
    
    # Plot Communication Cost if available
    if 'communication_cost' in df.columns:
        sns.lineplot(ax=axes[1, 1], data=df, x=df.index, y='communication_cost')
        axes[1, 1].set_title('Communication Cost Over Rounds')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Cost (bytes)')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Communication Cost Data', ha='center', va='center')
        axes[1, 1].set_title('Communication Cost')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(log_file.replace('.csv', '.png'))
    plt.show()