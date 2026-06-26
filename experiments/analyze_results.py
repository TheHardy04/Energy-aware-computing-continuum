"""
analyze_results.py
Analyzes raw GCP and Storm telemetry to compute energy efficiency and latency metrics.
Crucially, this evaluation model is strictly isomorphic to the objective function 
used in the CSP/LLM placement algorithms (Node Energy + Link Energy).
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

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# ENERGY MODEL (Strictly mirroring cspPlacement.py & energy_calculus.py)
# ==============================================================================
POWER_PROFILES = {
    "cloud-core":        {"p_idle": 9.0, "p_max": 25.0},
    "fog-gateway":       {"p_idle": 4.5, "p_max": 12.5},
    "worker-edge-light": {"p_idle": 1.5, "p_max":  3.5},
    "worker-edge":       {"p_idle": 2.5, "p_max":  6.5},
    "worker-iot":        {"p_idle": 0.5, "p_max":  2.0},
}

NETWORK_ENERGY_PER_BYTE = 0.00005  # Base J/Byte
GCP_PUE = 1.09

def get_power_profile(vm_name):
    """Matches a VM name to its power profile based on its prefix."""
    for prefix, profile in POWER_PROFILES.items():
        if prefix in vm_name:
            return profile
    return {"p_idle": 5.0, "p_max": 15.0}

def get_link_factor(latency_ms):
    """
    Mirrors the link_factor_scaled function from the CSP placement.
    Penalizes traffic crossing zones or regions.
    """
    if latency_ms <= 2.0:
        return 1.0  # Intra-zone
    elif latency_ms <= 15.0:
        return 1.5  # Inter-zone
    else:
        return 2.0  # Cross-region

# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def load_and_filter_data():
    """Loads CSVs, converts timestamps, and filters the last N minutes."""
    if not os.path.exists(GCP_CSV) or not os.path.exists(STORM_CSV):
        raise FileNotFoundError(f"Missing CSV files in {RAW_DIR}/. Run the experiment first.")

    df_gcp = pd.read_csv(GCP_CSV)
    df_storm = pd.read_csv(STORM_CSV)

    df_gcp["Exact_GCP_Timestamp"] = pd.to_datetime(df_gcp["Exact_GCP_Timestamp"])
    df_storm["Timestamp"] = pd.to_datetime(df_storm["Timestamp"])

    max_time = min(df_gcp["Exact_GCP_Timestamp"].max(), df_storm["Timestamp"].max())
    start_time = max_time - timedelta(minutes=WINDOW_MINUTES)

    df_gcp_filtered = df_gcp[(df_gcp["Exact_GCP_Timestamp"] >= start_time) & (df_gcp["Exact_GCP_Timestamp"] <= max_time)].copy()
    df_storm_filtered = df_storm[(df_storm["Timestamp"] >= start_time) & (df_storm["Timestamp"] <= max_time)].copy()

    return df_gcp_filtered, df_storm_filtered, start_time, max_time

def analyze_performance(df_gcp, df_storm, duration_seconds):
    """Computes KPIs using the CSP-aligned energy mathematical model."""
    
    # 1. Applicative Metrics (Storm)
    raw_emitted_diff = df_storm["Tuples_Emitted"].max() - df_storm["Tuples_Emitted"].min()
    total_real_tuples = raw_emitted_diff / 2  # Adjusting for __acker messages
    throughput_tps = total_real_tuples / duration_seconds if duration_seconds > 0 else 0
    
    # We use Storm's real end-to-end latency as the topological penalty multiplier
    avg_latency_ms = df_storm["Average_Latency_ms"].mean()
    link_penalty_factor = get_link_factor(avg_latency_ms)

    # 2. Infrastructure Metrics (Re-calculating Power based on CSP Model)
    # We create new columns to hold the mathematically aligned power values
    df_gcp["CSP_Compute_Power_W"] = 0.0
    df_gcp["CSP_Network_Power_W"] = 0.0

    interval_sec = 60.0 # Assuming standard 60s sampling for bandwidth rate

    for idx, row in df_gcp.iterrows():
        vm_name = row["VM_Name"]
        cpu_usage = row["CPU_Usage_Percent"]
        rx_tx_bytes = row["Network_RX_Bytes"] + row["Network_TX_Bytes"]
        
        # A. Node Energy Model: P_idle + (P_max - P_idle) * CPU%
        profile = get_power_profile(vm_name)
        p_cpu = profile["p_idle"] + (profile["p_max"] - profile["p_idle"]) * (cpu_usage / 100.0)
        
        # B. Link Energy Model: Bandwidth * Base_Energy * Latency * Link_Factor
        bytes_per_sec = rx_tx_bytes / interval_sec
        p_net = bytes_per_sec * NETWORK_ENERGY_PER_BYTE * avg_latency_ms * link_penalty_factor
        
        # Apply Data Center Power Usage Effectiveness (PUE)
        df_gcp.at[idx, "CSP_Compute_Power_W"] = p_cpu * GCP_PUE
        df_gcp.at[idx, "CSP_Network_Power_W"] = p_net * GCP_PUE

    # Aggregate cluster-wide power at each timestamp
    cluster_power_over_time = df_gcp.groupby("Exact_GCP_Timestamp").agg({
        "CSP_Compute_Power_W": "sum",
        "CSP_Network_Power_W": "sum"
    }).reset_index()
    
    cluster_power_over_time["Total_Power_W"] = cluster_power_over_time["CSP_Compute_Power_W"] + cluster_power_over_time["CSP_Network_Power_W"]

    avg_compute_power_w = cluster_power_over_time["CSP_Compute_Power_W"].mean()
    avg_network_power_w = cluster_power_over_time["CSP_Network_Power_W"].mean()
    avg_total_power_w = cluster_power_over_time["Total_Power_W"].mean()

    total_energy_joules = avg_total_power_w * duration_seconds
    joules_per_tuple = total_energy_joules / total_real_tuples if total_real_tuples > 0 else 0

    metrics = {
        "Duration_Seconds": duration_seconds,
        "Total_Tuples_Processed": total_real_tuples,
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
    """Generates a side-by-side plot matching academic paper styles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Response Time
    ax1.plot(df_storm["Timestamp"], df_storm["Average_Latency_ms"], color="tab:blue", marker="o", markersize=4, linestyle="-", label="Complete Latency")
    ax1.set_title("Response Time Stability (Last 15 Mins)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Response Time (ms)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Cluster Power (Compute + CSP Network Model)
    ax2.plot(cluster_power_over_time["Exact_GCP_Timestamp"], cluster_power_over_time["Total_Power_W"], color="tab:red", linewidth=2, label="Total Power (CSP Model)")
    ax2.fill_between(cluster_power_over_time["Exact_GCP_Timestamp"], cluster_power_over_time["CSP_Compute_Power_W"], color="tab:orange", alpha=0.3, label="Compute Power")
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
            print("⚠️ Not enough data points in the last 15 minutes.")
            return

        metrics, cluster_power = analyze_performance(df_gcp, df_storm, duration_seconds)

        print("=====================================================")
        print("🔬 ALGOTEL COMPARATIVE ANALYSIS (LAST 15 MINS)")
        print("=====================================================")
        print(f"Time Window       : {start_time.strftime('%H:%M:%S')} to {max_time.strftime('%H:%M:%S')}")
        print(f"Throughput        : {metrics['Throughput_TPS']:.2f} real tuples/sec")
        print(f"Average Latency   : {metrics['Average_Latency_ms']:.2f} ms")
        print("-----------------------------------------------------")
        print(f"Avg Compute Power : {metrics['Compute_Power_W']:.2f} W (CSP Model)")
        print(f"Avg Network Power : {metrics['Network_Power_W']:.2f} W (CSP Model)")
        print(f"Total Cluster Pwr : {metrics['Avg_Cluster_Power_W']:.2f} W")
        print("-----------------------------------------------------")
        print(f"Total Energy Used : {metrics['Total_Energy_Joules']:.2f} Joules")
        print(f"Energy Efficiency : {metrics['Energy_Joules_Per_Tuple']:.4f} Joules/Tuple")
        print("=====================================================")

        # Save to summary CSV
        summary_file = os.path.join(RESULTS_DIR, "summary_kpi_comparison.csv")
        df_metrics = pd.DataFrame([metrics])
        header = not os.path.exists(summary_file)
        df_metrics.to_csv(summary_file, mode='a', header=header, index=False)
        
        plot_academic_graphs(df_storm, cluster_power)

    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

if __name__ == "__main__":
    main()