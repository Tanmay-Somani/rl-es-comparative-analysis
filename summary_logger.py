import os
import pandas as pd
import matplotlib.pyplot as plt

def summarize_and_save(log_file_path, experiment_name):
    """
    Reads a CSV log file, calculates detailed statistics, prints a summary,
    and saves both the summary and a performance plot to the 'results' directory.

    Args:
        log_file_path (str): The full path to the input CSV log file.
        experiment_name (str): The name of the experiment for titling outputs.
    """
    results_dir = "results"

    # --- 1. Validate Log File ---
    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found at '{log_file_path}'. Cannot generate summary.")
        return

    # --- 2. Ensure Results Directory Exists ---
    os.makedirs(results_dir, exist_ok=True)

    try:
        # --- 3. Read Data Using Pandas ---
        df = pd.read_csv(log_file_path)
        if df.empty:
            print(f"Warning: Log file '{log_file_path}' is empty. Skipping summary.")
            return

        # --- 4. Generate and Save Statistical Summary ---
        # .describe() provides a comprehensive statistical overview
        summary = df.describe()

        print(f"\n--- Statistical Summary for: {experiment_name} ---")
        print(summary)

        # Save the detailed summary to a text file for better readability
        summary_path = os.path.join(results_dir, f"{experiment_name}_summary.txt")
        summary.to_csv(summary_path, sep='\t')
        print(f"Success: Full summary saved to: {summary_path}")

        # --- 5. Generate and Save Performance Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Dynamically get column names for plotting
        x_axis_label = df.columns[0]
        y_axis_label = df.columns[1]

        # Plot the raw data points
        ax.plot(df[x_axis_label], df[y_axis_label], marker='o', linestyle='-', markersize=3, alpha=0.6, label='Raw Score')

        # Calculate and plot a moving average to show the trend
        if len(df) >= 10:
            window_size = max(10, len(df) // 10)  # Use a 10% window or 10, whichever is larger
            moving_avg = df[y_axis_label].rolling(window=window_size).mean()
            ax.plot(df[x_axis_label], moving_avg, linestyle='--', color='red', linewidth=2, label=f'{window_size}-Period Moving Avg')
            ax.legend()

        ax.set_title(f"Performance Trend for: {experiment_name.replace('_', ' ').title()}", fontsize=16)
        ax.set_xlabel(x_axis_label.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_axis_label.replace('_', ' ').title(), fontsize=12)
        ax.grid(True) # Ensure grid is visible

        plot_path = os.path.join(results_dir, f"{experiment_name}_performance.png")
        plt.savefig(plot_path)
        plt.close(fig)  # Important: close the figure to free up memory

        print(f"Success: Performance plot saved to: {plot_path}")

    except Exception as e:
        print(f"Error: Could not process log file {log_file_path}. Reason: {e}")