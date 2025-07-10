# analysis_plotter.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
LOGS_DIR = "logs"
RESULTS_DIR = "results"
EXPERIMENTS = ["centralized_rl", "federated_rl", "centralized_es", "federated_es"]

# (The generate_single_report function remains the same)
def generate_single_report(log_file_path, experiment_name):
    # ... This function's code is unchanged ...
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
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[ERROR] processing {log_file_path}: {e}")

# --- Helper function for loading and normalizing all data ---
def load_all_data():
    all_data = {}
    for exp_name in EXPERIMENTS:
        log_path = os.path.join(LOGS_DIR, exp_name, "training_log.csv")
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            if df.empty: continue
            metric_col = df.columns[1]
            min_val, max_val = df[metric_col].min(), df[metric_col].max()
            df['normalized_performance'] = 0.5 if (max_val - min_val) == 0 else (df[metric_col] - min_val) / (max_val - min_val)
            all_data[exp_name] = df
        else:
             print(f"[WARNING] Log for '{exp_name}' not found.")
    return all_data

# (The generate_comparison_reports and time-based reports are unchanged)
def generate_comparison_reports(all_data):
    # ... This function's code is unchanged ...
    comparison_results_dir = os.path.join(RESULTS_DIR, "comparison_reports")
    os.makedirs(comparison_results_dir, exist_ok=True)
    print("\n>>> Generating Performance vs. Iteration comparison plots...")
    plot_definitions = {
        "Performance vs. Iterations (Centralized)": (["centralized_rl", "centralized_es"], "iter_comparison_centralized.png"),
        "Performance vs. Iterations (Federated)": (["federated_rl", "federated_es"], "iter_comparison_federated.png"),
        "Overall Performance vs. Iterations": (EXPERIMENTS, "iter_comparison_all_methods.png")
    }
    for title, (exp_list, filename) in plot_definitions.items():
        plt.figure(figsize=(14, 8))
        for name in exp_list:
            if name in all_data:
                df = all_data[name]
                window_size = max(10, len(df) // 20)
                smooth_perf = df['normalized_performance'].rolling(window_size, center=True).mean()
                plt.plot(df.index, smooth_perf, label=name.replace('_', ' ').title(), linewidth=2.5)
        plt.title(title, fontsize=18, weight='bold'); plt.xlabel("Training Iteration", fontsize=14); plt.ylabel("Normalized Performance", fontsize=14); plt.legend(fontsize=12); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(comparison_results_dir, filename), dpi=150); plt.close()
        print(f"[SUCCESS] Saved plot: {filename}")

def generate_time_based_reports(all_data):
    # ... This function's code is unchanged ...
    comparison_results_dir = os.path.join(RESULTS_DIR, "comparison_reports")
    os.makedirs(comparison_results_dir, exist_ok=True)
    print("\n>>> Generating Communication Cost (Wall-Clock Time) plots...")
    if not any('elapsed_time' in df.columns for df in all_data.values()): return
    plot_definitions = {
        "Communication Cost: Centralized vs. Federated RL": (["centralized_rl", "federated_rl"], "time_comparison_rl.png"),
        "Communication Cost: Centralized vs. Federated ES": (["centralized_es", "federated_es"], "time_comparison_es.png")
    }
    for title, (exp_list, filename) in plot_definitions.items():
        plt.figure(figsize=(14, 8))
        for name in exp_list:
            if name in all_data and 'elapsed_time' in all_data[name].columns:
                df = all_data[name]
                plt.plot(df['elapsed_time'], df['normalized_performance'], label=name.replace('_', ' ').title(), linewidth=2.5)
        plt.title(title, fontsize=18, weight='bold'); plt.xlabel("Wall-Clock Time (s)", fontsize=14); plt.ylabel("Normalized Performance", fontsize=14); plt.legend(fontsize=12); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(comparison_results_dir, filename), dpi=150); plt.close()
        print(f"[SUCCESS] Saved plot: {filename}")


### NEW: Function to generate the Privacy vs. Performance vs. Cost plot ###
def generate_privacy_tradeoff_plot(all_data, target_performance=0.8):
    """Generates a scatter plot visualizing the trade-off between performance, time, and privacy."""
    comparison_results_dir = os.path.join(RESULTS_DIR, "comparison_reports")
    os.makedirs(comparison_results_dir, exist_ok=True)
    print(f"\n>>> Generating Privacy vs. Performance Trade-off plot...")

    summary_data = []
    for name, df in all_data.items():
        try:
            # Assign Privacy Score based on architecture
            privacy_score = 1.0 if "federated" in name else 0.0
            architecture = "Federated" if privacy_score == 1.0 else "Centralized"
            algorithm = "ES" if "es" in name else "RL"
            
            # Calculate final performance (average of last 10% of run)
            final_performance = df['normalized_performance'].tail(len(df) // 10).mean()

            # Calculate time to reach target performance
            converged_df = df[df['normalized_performance'] >= target_performance]
            time_to_converge = converged_df.iloc[0]['elapsed_time'] if not converged_df.empty and 'elapsed_time' in df.columns else float('nan')

            summary_data.append({
                'Method': name.replace('_', ' ').title(),
                'Architecture': architecture,
                'Algorithm': algorithm,
                'Final Performance': final_performance,
                'Time to Converge (s)': time_to_converge
            })
        except Exception as e:
            print(f"[WARNING] Could not compute summary for {name}: {e}")
            continue
            
    if not summary_data:
        print("[WARNING] No data available for privacy trade-off plot.")
        return

    summary_df = pd.DataFrame(summary_data)
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    plot = sns.scatterplot(
        data=summary_df,
        x='Final Performance',
        y='Time to Converge (s)',
        hue='Architecture',
        style='Algorithm',
        s=200, # size of markers
        palette={'Centralized': 'crimson', 'Federated': 'royalblue'},
        edgecolor='black'
    )
    
    # Add annotations
    for i in range(summary_df.shape[0]):
        plt.text(
            x=summary_df['Final Performance'][i] + 0.01, 
            y=summary_df['Time to Converge (s)'][i], 
            s=summary_df['Method'][i],
            fontdict=dict(color='black', size=10)
        )
    
    plt.title('Performance vs. Efficiency Trade-off: The Cost of Privacy', fontsize=18, weight='bold')
    plt.xlabel('Final Model Performance (Normalized)', fontsize=14)
    plt.ylabel('Time to Reach 80% Performance (s)', fontsize=14)
    plt.legend(title='Architecture / Algorithm', fontsize=12)
    plt.xlim(summary_df['Final Performance'].min() - 0.05, summary_df['Final Performance'].max() + 0.15)
    plt.ylim(0, summary_df['Time to Converge (s)'].max() * 1.1)
    
    save_path = os.path.join(comparison_results_dir, "summary_privacy_tradeoff.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved plot: summary_privacy_tradeoff.png")


# --- Main CLI Execution ---
def main():
    # ... (parser definition is unchanged) ...
    parser = argparse.ArgumentParser(description="Tool for analyzing and plotting grid world experiment results.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", choices=["single", "compare"], help="""Choose the operation mode:\n'single':  Report for one experiment's log file.\n'compare': Compare all available experiment logs.""")
    parser.add_argument("--log_file", help="Path to the log file (for 'single' mode).")
    parser.add_argument("--name", help="Name of the experiment (for 'single' mode).")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.log_file or not args.name:
            parser.error("--log_file and --name are required for 'single' mode.")
        generate_single_report(args.log_file, args.name)
    elif args.mode == "compare":
        all_data = load_all_data()
        if not all_data:
            print("[ERROR] No log files found. Please run experiments first.")
            return
        # --- Call all comparison functions ---
        generate_comparison_reports(all_data)
        generate_time_based_reports(all_data)
        generate_privacy_tradeoff_plot(all_data) # NEW: Call the new function

if __name__ == "__main__":
    main()