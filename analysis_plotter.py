# analysis_plotter.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# --- Configuration ---
LOGS_DIR = "logs"
RESULTS_DIR = "results"
EXPERIMENTS = ["centralized_rl", "federated_rl", "centralized_es", "federated_es"]

def generate_single_report(log_file_path, experiment_name):
    """Reads a single CSV log, calculates stats, and saves a summary and plot."""
    single_results_dir = os.path.join(RESULTS_DIR, "single_reports")
    os.makedirs(single_results_dir, exist_ok=True)
    try:
        df = pd.read_csv(log_file_path)
        if df.empty: return
        summary = df.describe()
        summary_path = os.path.join(single_results_dir, f"{experiment_name}_summary.txt")
        summary.to_csv(summary_path, sep='\t')
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        x_axis_label, y_axis_label = df.columns[0], df.columns[1]
        sns.lineplot(x=x_axis_label, y=y_axis_label, data=df, ax=ax, marker='o', markersize=4, alpha=0.7, label='Raw Score')
        if len(df) >= 10:
            window_size = max(10, len(df) // 20)
            moving_avg = df[y_axis_label].rolling(window=window_size, center=True).mean()
            ax.plot(df[x_axis_label], moving_avg, color='crimson', linestyle='--', linewidth=2.5, label=f'{window_size}-Period Moving Avg')
            ax.legend()
        title_name = experiment_name.replace('_', ' ').title()
        ax.set_title(f"Performance Trend: {title_name}", fontsize=16, weight='bold')
        ax.set_xlabel(x_axis_label.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_axis_label.replace('_', ' ').title(), fontsize=12)
        plot_path = os.path.join(single_results_dir, f"{experiment_name}_performance.png")
        # MODIFIED: Added plt.tight_layout() for robust fitting
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[ERROR] processing {log_file_path}: {e}")

def load_and_normalize_data(run_files):
    """Loads a list of log files, normalizes their performance, and returns a list of dataframes."""
    data_frames = []
    for file in run_files:
        df = pd.read_csv(file)
        if df.empty: continue
        metric_col = df.columns[1]
        min_val, max_val = df[metric_col].min(), df[metric_col].max()
        df['normalized_performance'] = 0.5 if (max_val - min_val) == 0 else (df[metric_col] - min_val) / (max_val - min_val)
        data_frames.append(df)
    return data_frames

def generate_robustness_reports(all_data_runs):
    """Generates plots analyzing the robustness of algorithms across multiple runs."""
    robustness_results_dir = os.path.join(RESULTS_DIR, "robustness_reports")
    os.makedirs(robustness_results_dir, exist_ok=True)
    print("\n>>> Generating Robustness analysis plots...")

    plt.figure(figsize=(16, 9))
    final_scores = []
    
    for name, df_list in all_data_runs.items():
        if not df_list: continue
        # Concatenate runs with a new 'run' key for seaborn
        for i, df in enumerate(df_list):
            df['run'] = i
        
        all_runs_df = pd.concat(df_list)
        x_col, y_col = all_runs_df.columns[0], all_runs_df.columns[1]

        # Plot confidence interval directly using seaborn
        sns.lineplot(data=all_runs_df, x=x_col, y=y_col, label=name.replace('_', ' ').title(), errorbar='sd')

        # Collect final scores for box plot
        for df in df_list:
            final_score = df[y_col].tail(len(df) // 20).mean()
            final_scores.append({'Method': name.replace('_', ' ').title(), 'Final Score': final_score})

    plt.title('Algorithm Robustness: Mean Performance with Confidence Interval', fontsize=18, weight='bold')
    plt.xlabel('Training Iteration', fontsize=14)
    plt.ylabel('Performance Metric (Non-normalized)', fontsize=14)
    plt.legend(fontsize=12); plt.grid(True, which='both', linestyle='--'); plt.tight_layout()
    # MODIFIED: Added bbox_inches='tight' for robust saving
    plt.savefig(os.path.join(robustness_results_dir, "robustness_confidence_interval.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Box Plot of Final Performance
    if final_scores:
        final_scores_df = pd.DataFrame(final_scores)
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='Final Score', y='Method', data=final_scores_df, palette="coolwarm", orient='h')
        sns.stripplot(x='Final Score', y='Method', data=final_scores_df, color='black', alpha=0.5, jitter=0.1)
        plt.title('Distribution of Final Performance Across Multiple Runs', fontsize=18, weight='bold')
        plt.xlabel('Final Performance Score', fontsize=14); plt.ylabel('')
        plt.tight_layout()
        # MODIFIED: Added bbox_inches='tight' for robust saving
        plt.savefig(os.path.join(robustness_results_dir, "robustness_final_performance_boxplot.png"), dpi=150, bbox_inches='tight')
        plt.close()

def generate_comparison_plots(all_data):
    """Generates comparison plots for performance, time, and privacy trade-offs."""
    comparison_results_dir = os.path.join(RESULTS_DIR, "comparison_reports")
    os.makedirs(comparison_results_dir, exist_ok=True)
    print("\n>>> Generating all comparison plots...")

    # Plot 1: Performance vs. Time (Communication Cost)
    plt.figure(figsize=(14, 8))
    for name, df in all_data.items():
        if 'elapsed_time' in df.columns:
            plt.plot(df['elapsed_time'], df['normalized_performance'], label=name.replace('_', ' ').title(), linewidth=2.5)
    plt.title('Communication Cost: Performance vs. Wall-Clock Time', fontsize=18, weight='bold')
    plt.xlabel('Wall-Clock Time (seconds)', fontsize=14); plt.ylabel('Normalized Performance (0 to 1)', fontsize=14)
    plt.legend(fontsize=12); plt.grid(True); plt.tight_layout()
    # MODIFIED: Added bbox_inches='tight' for robust saving
    plt.savefig(os.path.join(comparison_results_dir, "comparison_time.png"), dpi=150, bbox_inches='tight'); plt.close()

    # Plot 2: Privacy Trade-off Scatter Plot
    summary_data = []
    for name, df in all_data.items():
        try:
            privacy_score = 1.0 if "federated" in name else 0.0
            final_performance = df['normalized_performance'].tail(len(df) // 10).mean()
            time_to_converge = df[df['normalized_performance'] >= 0.8].iloc[0]['elapsed_time'] if not df[df['normalized_performance'] >= 0.8].empty else float('nan')
            summary_data.append({'Method': name.replace('_', ' ').title(), 'Architecture': 'Federated' if privacy_score else 'Centralized', 'Algorithm': 'ES' if 'es' in name else 'RL', 'Final Performance': final_performance, 'Time to Converge (s)': time_to_converge})
        except Exception: continue
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=summary_df, x='Final Performance', y='Time to Converge (s)', hue='Architecture', style='Algorithm', s=200, palette={'Centralized': 'crimson', 'Federated': 'royalblue'}, edgecolor='black')
        for i in range(summary_df.shape[0]):
            plt.text(x=summary_df['Final Performance'][i]+0.01, y=summary_df['Time to Converge (s)'][i], s=summary_df['Method'][i])
        plt.title('Performance vs. Efficiency: The Cost of Privacy', fontsize=18, weight='bold'); plt.xlabel('Final Model Performance', fontsize=14); plt.ylabel('Time to Reach 80% Performance (s)', fontsize=14)
        plt.legend(title='Architecture / Algorithm', fontsize=12); plt.tight_layout()
        # MODIFIED: Added bbox_inches='tight' for robust saving
        plt.savefig(os.path.join(comparison_results_dir, "summary_privacy_tradeoff.png"), dpi=150, bbox_inches='tight'); plt.close()

def main():
    parser = argparse.ArgumentParser(description="Grid World Experiment Analysis Tool", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", choices=["single", "compare"], help="Choose operation mode.")
    parser.add_argument("--log_file", help="Path to log file (for 'single' mode).")
    parser.add_argument("--name", help="Name of experiment (for 'single' mode).")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.log_file or not args.name: parser.error("--log_file and --name are required.")
        generate_single_report(args.log_file, args.name)
    elif args.mode == "compare":
        all_data_runs = {name: load_and_normalize_data(glob(os.path.join(LOGS_DIR, name, "*.csv"))) for name in EXPERIMENTS}
        all_single_data = {name: df_list[0] for name, df_list in all_data_runs.items() if df_list}
        
        if all_single_data: generate_comparison_plots(all_single_data)
        else: print("[WARNING] No single run logs found for standard comparison plots.")
        
        if any(len(v) > 1 for v in all_data_runs.values()): generate_robustness_reports(all_data_runs)
        else: print("[WARNING] No multi-run data found for robustness plots.")

if __name__ == "__main__":
    main()