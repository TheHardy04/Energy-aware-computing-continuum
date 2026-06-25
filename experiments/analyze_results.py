"""
analyze_results.py
Analyzes raw GCP and Storm telemetry to compute energy efficiency and latency metrics
over the last 15 minutes of the experiment, generating academic-style plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# --- Configuration ---
RAW_DIR = "experiments/raw"
RESULTS_DIR = "experiments/results"
GCP_CSV = os.path.join(RAW_DIR, "gcp_metrics.csv")
STORM_CSV = os.path.join(RAW_DIR, "storm_metrics.csv")
WINDOW_MINUTES = 15

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_filter_data():
    """Loads CSVs, converts timestamps, and filters the last 15 minutes."""
    if not os.path.exists(GCP_CSV) or not os.path.exists(STORM_CSV):
        raise FileNotFoundError(f"Missing CSV files in {RAW_DIR}/. Run the experiment first.")

    # Load data
    df_gcp = pd.read_csv(GCP_CSV)
    df_storm = pd.read_csv(STORM_CSV)

    # Convert timestamps to datetime objects
    df_gcp["Exact_GCP_Timestamp"] = pd.to_datetime(df_gcp["Exact_GCP_Timestamp"])
    df_storm["Timestamp"] = pd.to_datetime(df_storm["Timestamp"])

    # Find the latest timestamp across both datasets to define the end of the experiment
    max_time = min(df_gcp["Exact_GCP_Timestamp"].max(), df_storm["Timestamp"].max())
    start_time = max_time - timedelta(minutes=WINDOW_MINUTES)

    # Filter data for the last WINDOW_MINUTES
    df_gcp_filtered = df_gcp[(df_gcp["Exact_GCP_Timestamp"] >= start_time) & (df_gcp["Exact_GCP_Timestamp"] <= max_time)].copy()
    df_storm_filtered = df_storm[(df_storm["Timestamp"] >= start_time) & (df_storm["Timestamp"] <= max_time)].copy()

    return df_gcp_filtered, df_storm_filtered, start_time, max_time

def analyze_performance(df_gcp, df_storm, duration_seconds):
    """Computes KPIs: Average Latency, Throughput, and Total Energy."""
    
    # 1. Applicative Metrics (Storm)
    # Assuming 'Tuples_Emitted' is cumulative. Delta = Max - Min.
    total_tuples = df_storm["Tuples_Emitted"].max() - df_storm["Tuples_Emitted"].min()
    throughput_tps = total_tuples / duration_seconds if duration_seconds > 0 else 0
    avg_latency_ms = df_storm["Average_Latency_ms"].mean()

    # 2. Infrastructure Metrics (GCP)
    # Group by timestamp to get cluster-wide power at each recorded moment
    cluster_power_over_time = df_gcp.groupby("Exact_GCP_Timestamp").agg({
        "Compute_Power_W": "sum",
        "Network_Power_W": "sum",
        "Total_Power_W": "sum"
    }).reset_index()

    # Average power over the 15-minute window
    avg_total_power_w = cluster_power_over_time["Total_Power_W"].mean()
    avg_compute_power_w = cluster_power_over_time["Compute_Power_W"].mean()
    avg_network_power_w = cluster_power_over_time["Network_Power_W"].mean()

    # Total Energy (Joules) = Average Power (Watts) * Time (seconds)
    total_energy_joules = avg_total_power_w * duration_seconds

    # Energy Efficiency
    joules_per_tuple = total_energy_joules / total_tuples if total_tuples > 0 else 0

    metrics = {
        "Duration_Seconds": duration_seconds,
        "Total_Tuples_Processed": total_tuples,
        "Throughput_TPS": throughput_tps,
        "Average_Latency_ms": avg_latency_ms,
        "Avg_Cluster_Power_W": avg_total_power_w,
        "Compute_Power_W": avg_compute_power_w,
        "Network_Power_W": avg_network_power_w,
        "Total_Energy_Joules": total_energy_joules,
        "Energy_Joules_Per_Tuple": joules_per_tuple
    }

    return metrics, cluster_power_over_time

def plot_academic_graphs(df_storm, cluster_power_over_time):
    """Generates a side-by-side plot matching academic paper styles (like AlgoTel)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Response Time (Latency) over time
    ax1.plot(df_storm["Timestamp"], df_storm["Average_Latency_ms"], color="tab:blue", marker="o", markersize=4, linestyle="-", label="Complete Latency")
    ax1.set_title("Response Time Stability (Last 15 Mins)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Response Time (ms)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Cluster Power over time (Compute vs Network)
    ax2.plot(cluster_power_over_time["Exact_GCP_Timestamp"], cluster_power_over_time["Total_Power_W"], color="tab:red", linewidth=2, label="Total Power")
    ax2.fill_between(cluster_power_over_time["Exact_GCP_Timestamp"], cluster_power_over_time["Compute_Power_W"], color="tab:orange", alpha=0.3, label="Compute Power")
    ax2.set_title("Cluster Power Consumption")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Power (Watts)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "algo_comparison_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"📊 Plot saved to {plot_path}")

def main():
    try:
        df_gcp, df_storm, start_time, max_time = load_and_filter_data()
        duration_seconds = (max_time - start_time).total_seconds()

        if df_gcp.empty or df_storm.empty:
            print("⚠️ Not enough data points in the last 15 minutes. Did the experiment run long enough?")
            return

        metrics, cluster_power = analyze_performance(df_gcp, df_storm, duration_seconds)

        # Print standard console output for easy reading
        print("=====================================================")
        print("🔬 ALGOTEL COMPARATIVE ANALYSIS (LAST 15 MINS)")
        print("=====================================================")
        print(f"Time Window       : {start_time.strftime('%H:%M:%S')} to {max_time.strftime('%H:%M:%S')}")
        print(f"Throughput        : {metrics['Throughput_TPS']:.2f} tuples/sec")
        print(f"Average Latency   : {metrics['Average_Latency_ms']:.2f} ms")
        print("-----------------------------------------------------")
        print(f"Avg Compute Power : {metrics['Compute_Power_W']:.2f} W")
        print(f"Avg Network Power : {metrics['Network_Power_W']:.2f} W")
        print(f"Total Cluster Pwr : {metrics['Avg_Cluster_Power_W']:.2f} W")
        print("-----------------------------------------------------")
        print(f"Total Energy Used : {metrics['Total_Energy_Joules']:.2f} Joules")
        print(f"Energy Efficiency : {metrics['Energy_Joules_Per_Tuple']:.4f} Joules/Tuple")
        print("=====================================================")

        # Save metrics to a CSV line for multi-algorithm comparison
        # (Allows you to append Greedy, LLM, CSP runs into one file later)
        summary_file = os.path.join(RESULTS_DIR, "summary_kpi_comparison.csv")
        df_metrics = pd.DataFrame([metrics])
        header = not os.path.exists(summary_file)
        df_metrics.to_csv(summary_file, mode='a', header=header, index=False)
        
        plot_academic_graphs(df_storm, cluster_power)

    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

if __name__ == "__main__":
    main()