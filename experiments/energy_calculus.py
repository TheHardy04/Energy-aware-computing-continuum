from pathlib import Path
import pandas as pd

# ==============================================================================
# 1. POWER PROFILES CONFIGURATION (Based on Cloud Carbon Footprint & SPECpower)
# 
# Hardware baseline assumptions (GCP Medians):
# - 1 vCPU (GCP Median) = 0.71 W (Idle) to 4.26 W (Max)
# - 1 GB RAM (DDR4) = ~0.38 W (Active/Idle average)
# Format: "vm_prefix": {"p_idle": Watts, "p_max": Watts}
# ==============================================================================
POWER_PROFILES = {
    "cloud-core":        {"p_idle": 9.0, "p_max": 25.0},  # n2-standard-4 (4 vCPU, 16GB)
    "fog-gateway":       {"p_idle": 4.5, "p_max": 12.5},  # e2-standard-2 (2 vCPU, 8GB)
    "worker-edge-light": {"p_idle": 1.5, "p_max": 3.5},   # e2-small (Shared core, 2GB)
    "worker-edge":       {"p_idle": 2.5, "p_max": 6.5},   # e2-medium (Shared core, 4GB)
    "worker-iot":        {"p_idle": 0.5, "p_max": 2.0}    # e2-micro (Shared core, 1GB)
}

# Network energy factor (Joules per Byte transferred)
# 0.00005 J/Byte = 50 kJ / GB (Average Edge/Cloud IP core routing footprint)
NETWORK_ENERGY_PER_BYTE = 0.00005

# Time between each metric sample in your data (in seconds)
TIME_INTERVAL_SEC = 60.0

# ==============================================================================
# 2. GOOGLE ENVIRONMENTAL REPORT DATA (2024/2025 values)
# ==============================================================================
# Global average Power Usage Effectiveness for Google Data Centers
GCP_PUE = 1.09 

# Grid Carbon Intensity (Grams of CO2eq per kWh)
# Based on the Carbon-Free Energy (CFE) percentage per region.
CARBON_INTENSITY_G_CO2_KWH = {
    "europe-west9": 50.0,   # Europe / France (Very high CFE / Nuclear+Renewables)
    "us-central1":  350.0,  # North America (Average CFE)
    "asia-east1":   500.0,  # Asia-Pacific (Low CFE)
    "default":      250.0
}

# Set the region where the experiment was running
EXPERIMENT_REGION = "europe-west9"

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def get_power_profile(vm_name):
    """Matches a VM name to its power profile based on its prefix."""
    for prefix, profile in POWER_PROFILES.items():
        if prefix in vm_name:
            return profile
    # Default profile if not found
    return {"p_idle": 5.0, "p_max": 15.0}

def calculate_energy(data_source):
    # Load metrics data
    df = pd.read_csv(data_source)

    # --------------------------------------------------------------------------
    # DATA CLEANING: Filter out GCP API failures
    # A live VM will never have exactly 0.0 CPU, 0.0 TX Bytes, and 0.0 RX Bytes.
    # --------------------------------------------------------------------------
    df = df[~(
        (df['CPU_Usage_Percent'] == 0.0) & 
        (df['Network_TX_Bytes'] == 0.0) & 
        (df['Network_RX_Bytes'] == 0.0)
    )].copy()

    # Lists to store computed values
    cpu_energy_j = []
    net_energy_j = []

    for index, row in df.iterrows():
        vm_name = row['VM_Name']
        cpu_usage = row['CPU_Usage_Percent']
        total_bytes = row['Network_RX_Bytes'] + row['Network_TX_Bytes']

        profile = get_power_profile(vm_name)

        # 1. Calculate CPU power at this timestamp (Watts)
        p_cpu = profile['p_idle'] + (profile['p_max'] - profile['p_idle']) * (cpu_usage / 100.0)

        # 2. Calculate CPU energy over the elapsed interval (Joules = Watts * Seconds)
        e_cpu_j = p_cpu * TIME_INTERVAL_SEC

        # 3. Calculate Network energy (Joules)
        e_net_j = total_bytes * NETWORK_ENERGY_PER_BYTE

        cpu_energy_j.append(e_cpu_j)
        net_energy_j.append(e_net_j)

    # Append IT energy columns (Server + Network)
    df['Energy_CPU_Joules'] = cpu_energy_j
    df['Energy_Net_Joules'] = net_energy_j
    df['Energy_IT_Total_Joules'] = df['Energy_CPU_Joules'] + df['Energy_Net_Joules']
    
    # 4. Apply GCP's PUE to get the total energy drawn from the power grid
    df['Energy_Grid_Total_Joules'] = df['Energy_IT_Total_Joules'] * GCP_PUE
    
    # Convert Joules to Watt-hours (1 Wh = 3600 Joules)
    df['Energy_Grid_Total_Wh'] = df['Energy_Grid_Total_Joules'] / 3600.0

    # 5. Calculate Carbon Footprint (Grams of CO2eq)
    # Formula: (Wh / 1000) * Carbon Intensity (gCO2/kWh)
    intensity = CARBON_INTENSITY_G_CO2_KWH.get(EXPERIMENT_REGION, CARBON_INTENSITY_G_CO2_KWH["default"])
    df['Carbon_Emissions_g'] = (df['Energy_Grid_Total_Wh'] / 1000.0) * intensity

    return df

def discover_metrics_files(results_dir):
    return sorted(results_dir.glob("*_gcp_experiment_metrics.csv"))

def strategy_name_from_path(file_path):
    return file_path.name.replace("_gcp_experiment_metrics.csv", "").upper()

def summarize_strategy(file_path):
    df_result = calculate_energy(file_path)
    
    # Aggregate metrics per VM
    vm_summary = df_result.groupby('VM_Name')[
        ['Energy_CPU_Joules', 'Energy_Net_Joules', 'Energy_Grid_Total_Joules', 'Energy_Grid_Total_Wh', 'Carbon_Emissions_g']
    ].sum()
    
    totals = vm_summary.sum()
    
    # We use the cleaned dataframe length, ensuring broken samples don't artificially increase duration
    sample_count = len(df_result)
    vm_count = df_result['VM_Name'].nunique()
    
    # Normalization: Calculate the effective duration of the experiment in minutes
    # (Total valid samples / Number of VMs) * (Interval in seconds / 60)
    experiment_duration_mins = (sample_count / vm_count) * (TIME_INTERVAL_SEC / 60.0)

    return {
        'strategy': strategy_name_from_path(file_path),
        'sample_count': sample_count,
        'vm_count': vm_count,
        'duration_mins': experiment_duration_mins,
        'total_cpu_joules': totals['Energy_CPU_Joules'],
        'total_net_joules': totals['Energy_Net_Joules'],
        'total_energy_grid_wh': totals['Energy_Grid_Total_Wh'],
        'total_carbon_g': totals['Carbon_Emissions_g'],
        # Normalized metrics for fair comparison across different execution times
        'normalized_wh_per_min': totals['Energy_Grid_Total_Wh'] / experiment_duration_mins,
        'normalized_carbon_per_min': totals['Carbon_Emissions_g'] / experiment_duration_mins,
        'vm_summary': vm_summary.sort_values('Energy_Grid_Total_Wh', ascending=False),
    }

def print_strategy_summary(summary):
    print(f"=== STRATEGY: {summary['strategy']} ===")
    print(
        f"Valid Duration: {summary['duration_mins']:.1f} mins | "
        f"Valid Samples: {summary['sample_count']} | "
        f"VMs: {summary['vm_count']}\n"
        f"Raw Totals -> Grid Energy: {summary['total_energy_grid_wh']:.4f} Wh | "
        f"Carbon: {summary['total_carbon_g']:.4f} g CO2eq\n"
        f"NORMALIZED -> {summary['normalized_wh_per_min']:.4f} Wh/min | "
        f"{summary['normalized_carbon_per_min']:.4f} g CO2eq/min"
    )
    print("\nPer-VM energy totals (Raw):")
    # Drop Joules column for cleaner display in the per-VM breakdown
    display_summary = summary['vm_summary'].drop(columns=['Energy_Grid_Total_Joules'])
    print(display_summary.round(4).to_string())
    print("-" * 60 + "\n")

def main():
    results_dir = Path(__file__).resolve().parent
    metrics_files = discover_metrics_files(results_dir)

    if not metrics_files:
        print(f"No strategy metrics files found in {results_dir}")
        return

    strategy_summaries = [summarize_strategy(file_path) for file_path in metrics_files]

    print("=== ENERGY AND CARBON SUMMARY PER STRATEGY ===\n")
    for summary in strategy_summaries:
        print_strategy_summary(summary)

    # Build the final comparison table focusing on NORMALIZED metrics
    totals_df = pd.DataFrame(
        [
            {
                'Strategy': summary['strategy'],
                'Duration(m)': round(summary['duration_mins'], 1),
                'Total_Wh': summary['total_energy_grid_wh'],
                'Total_CO2_g': summary['total_carbon_g'],
                'Norm_Wh/min': summary['normalized_wh_per_min'],
                'Norm_CO2/min': summary['normalized_carbon_per_min'],
            }
            for summary in strategy_summaries
        ]
    ).sort_values('Norm_Wh/min') # Sort by the fairest metric

    print("=== FAIR STRATEGY COMPARISON (Sorted by Normalized Energy) ===")
    print(totals_df.round(4).to_string(index=False))

if __name__ == '__main__':
    main()