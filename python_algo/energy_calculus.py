from pathlib import Path
import pandas as pd
import re

from src.gcpEnergyModel import _load_energy_settings, link_factor

# Power profiles by VM type (Watts). Based on GCP hardware + SPECpower data.
POWER_PROFILES = {
    "cloud-core":        {"p_idle": 9.0, "p_max": 25.0},  # n2-standard-4 (4 vCPU, 16 GB)
    "fog-gateway":       {"p_idle": 4.5, "p_max": 12.5},  # e2-standard-2 (2 vCPU, 8 GB)
    "worker-edge-light": {"p_idle": 1.5, "p_max":  3.5},  # e2-small (shared core, 2 GB)
    "worker-edge":       {"p_idle": 2.5, "p_max":  6.5},  # e2-medium (shared core, 4 GB)
    "worker-iot":        {"p_idle": 0.5, "p_max":  2.0},  # e2-micro (shared core, 1 GB)
}

# J/Byte for network traffic (≈ 0.014 kWh/GB — Aslan et al. 2017 and IEA)
NETWORK_ENERGY_PER_BYTE = 0.00005

# Seconds between metric samples
TIME_INTERVAL_SEC = 60.0

# Google environmental data (2024/2025)
GCP_PUE = 1.09

# gCO₂eq per kWh, by GCP region (derived from Carbon-Free Energy percentages)
CARBON_INTENSITY_G_CO2_KWH = {
    "europe-west9": 50.0,   # France — nuclear + renewables
    "us-central1":  350.0,  # North America — average grid
    "asia-east1":   500.0,  # Asia-Pacific — coal-heavy grid
    "default":      250.0,
}

EXPERIMENT_REGION = "europe-west9"

PROJECT_ROOT          = Path(__file__).resolve().parents[1]
INFRA_PROPERTIES_PATH = Path(__file__).resolve().parent / "properties" / "Infra_5nodes_GCP.properties"
INFRA_MAPPING_PATH    = Path(__file__).resolve().parent / "properties" / "Infra_5nodes_GCP_mapping.csv"
METRICS_RESULTS_DIR   = PROJECT_ROOT / "results"


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_power_profile(vm_name):
    """Return the idle/max power profile (Watts) for a VM, matched by name prefix."""
    for prefix, profile in POWER_PROFILES.items():
        if prefix in vm_name:
            return profile
    return {"p_idle": 5.0, "p_max": 15.0}  # generic fallback


def _join_continuation_lines(raw_text):
    """Merge Java-properties continuation lines ending with '\\'."""
    out, current = [], ""
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
        else:
            current += line
            out.append(current.strip())
            current = ""
    if current:
        out.append(current.strip())
    return out


def _zone_to_region(zone):
    """Convert 'europe-west9-a' → 'europe-west9'."""
    if not isinstance(zone, str) or not zone:
        return ""
    return re.sub(r"-[a-z]$", "", zone.strip())


def _load_infra5_metadata():
    """
    Parse Infra_5nodes_GCP.properties and its VM→host mapping CSV.

    Returns
    -------
    vm_zone_map    : dict[vm_name -> zone_string]
    vm_latency_map : dict[vm_name -> average_outgoing_latency_ms]
    """
    if not INFRA_PROPERTIES_PATH.exists() or not INFRA_MAPPING_PATH.exists():
        return {}, {}

    with open(INFRA_PROPERTIES_PATH, "r", encoding="utf-8") as f:
        lines = _join_continuation_lines(f.read())

    zones, topology_line = [], ""
    for line in lines:
        if line.startswith("hosts.zones"):
            _, value = line.split("=", 1)
            zones = [z.strip() for z in value.split(",") if z.strip()]
        elif line.startswith("network.topology"):
            _, topology_line = line.split("=", 1)

    # Average outgoing latency per host index
    host_latency: dict[int, list[float]] = {}
    for src, dst, _bw, lat in re.findall(
        r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*\}", topology_line
    ):
        if src != dst:
            host_latency.setdefault(int(src), []).append(float(lat))

    host_latency_fallback = {
        h: sum(lats) / len(lats) for h, lats in host_latency.items() if lats
    }

    mapping_df = pd.read_csv(INFRA_MAPPING_PATH)
    mapping_df.columns = [str(c).strip().lower() for c in mapping_df.columns]
    if "host" not in mapping_df.columns or "vm" not in mapping_df.columns:
        return {}, {}

    vm_zone_map, vm_latency_map = {}, {}
    for _, row in mapping_df.iterrows():
        host_i  = int(row["host"])
        vm_name = str(row["vm"]).strip()
        vm_zone_map[vm_name]    = zones[host_i] if host_i < len(zones) else ""
        vm_latency_map[vm_name] = host_latency_fallback.get(host_i)

    return vm_zone_map, vm_latency_map


# ── Energy calculation ─────────────────────────────────────────────────────────

def _aggregate_metrics(df):
    """
    Drop API failure rows, then collapse each VM to a single representative
    row using the median CPU, RX, and TX across all valid samples.

    Energy consumption and data transfer are expected to be constant for a
    given placement, so the median is the right estimator — it is also
    robust to the rare background-transfer bursts seen in raw GCP data.

    Returns a DataFrame with one row per VM and an extra Sample_Count column
    that records how many valid samples were collected for that VM.
    """
    bad = (
        (df["CPU_Usage_Percent"] == 0.0) &
        (df["Network_TX_Bytes"]  == 0.0) &
        (df["Network_RX_Bytes"]  == 0.0)
    )
    df = df[~bad]

    counts  = df.groupby("VM_Name").size().rename("Sample_Count")
    medians = df.groupby("VM_Name")[
        ["CPU_Usage_Percent", "Network_RX_Bytes", "Network_TX_Bytes"]
    ].median()

    return medians.join(counts).reset_index()


def _row_energy(row, cfg, vm_latency_map):
    """
    Compute CPU and network energy (Joules) for a single monitoring sample.

    CPU model     : linear interpolation between p_idle and p_max.
    Network model : bytes × NETWORK_ENERGY_PER_BYTE × latency × tier_factor
                    (mirrors the simulation's link energy formulation).
    """
    profile = get_power_profile(row["VM_Name"])
    p_cpu   = profile["p_idle"] + (profile["p_max"] - profile["p_idle"]) * (row["CPU_Usage_Percent"] / 100.0)
    e_cpu   = p_cpu * TIME_INTERVAL_SEC

    if "Path_Latency_ms" in row.index and pd.notna(row["Path_Latency_ms"]):
        latency_ms = float(row["Path_Latency_ms"])
    else:
        latency_ms = float(vm_latency_map.get(row["VM_Name"], cfg["gcp.lat.interzone_ms"]))

    total_bytes = row["Network_RX_Bytes"] + row["Network_TX_Bytes"]
    e_net = total_bytes * NETWORK_ENERGY_PER_BYTE * latency_ms * link_factor(latency_ms, cfg)

    return e_cpu, e_net


def _add_carbon_columns(df, vm_zone_map):
    """Append VM_Zone, VM_Region, Carbon_Intensity, and Carbon_Emissions columns."""
    df["VM_Zone"]   = df["VM_Name"].map(lambda vm: vm_zone_map.get(vm, ""))
    df["VM_Region"] = df["VM_Zone"].map(lambda z: _zone_to_region(z) or EXPERIMENT_REGION)
    df["Carbon_Intensity_gCO2_kWh"] = df["VM_Region"].map(
        lambda r: CARBON_INTENSITY_G_CO2_KWH.get(r, CARBON_INTENSITY_G_CO2_KWH["default"])
    )
    df["Carbon_Emissions_g"] = (df["Energy_Grid_Total_Wh"] / 1000.0) * df["Carbon_Intensity_gCO2_kWh"]
    return df


def calculate_energy(data_source):
    """
    Load a metrics CSV, compute total IT energy and carbon emissions per VM,
    and return an enriched DataFrame (one row per VM).

    Energy per interval is derived from the median CPU/RX/TX across all valid
    samples, then scaled by the actual sample count to obtain totals.
    """
    df  = pd.read_csv(data_source)
    cfg = _load_energy_settings()
    vm_zone_map, vm_latency_map = _load_infra5_metadata()

    df = _aggregate_metrics(df)

    # Energy per single interval (Joules) at the median operating point
    energy_rows = [_row_energy(row, cfg, vm_latency_map) for _, row in df.iterrows()]
    df["Energy_CPU_Joules"], df["Energy_Net_Joules"] = zip(*energy_rows)

    # Scale by sample count → total energy over the full experiment
    df["Energy_CPU_Joules"] *= df["Sample_Count"]
    df["Energy_Net_Joules"] *= df["Sample_Count"]

    df["Energy_IT_Total_Joules"]   = df["Energy_CPU_Joules"] + df["Energy_Net_Joules"]
    df["Energy_Grid_Total_Joules"] = df["Energy_IT_Total_Joules"] * GCP_PUE
    df["Energy_Grid_Total_Wh"]     = df["Energy_Grid_Total_Joules"] / 3600.0

    df = _add_carbon_columns(df, vm_zone_map)

    return df

# ── Reporting ──────────────────────────────────────────────────────────────────

def discover_metrics_files(results_dir):
    return sorted(results_dir.glob("*_gcp_experiment_metrics.csv"))


def strategy_name_from_path(file_path):
    return file_path.name.replace("_gcp_experiment_metrics.csv", "").upper()


def summarize_strategy(file_path):
    df = calculate_energy(file_path)

    # df already has one row per VM after aggregation
    vm_summary = df.set_index("VM_Name")[
        ["Energy_CPU_Joules", "Energy_Net_Joules", "Energy_Grid_Total_Joules",
         "Energy_Grid_Total_Wh", "Carbon_Emissions_g"]
    ].sort_values("Energy_Grid_Total_Wh", ascending=False)

    totals       = vm_summary.sum()
    vm_count     = len(df)
    sample_count = int(df["Sample_Count"].sum())
    # Average samples per VM × interval gives the effective experiment duration
    duration_min = df["Sample_Count"].mean() * (TIME_INTERVAL_SEC / 60.0)

    return {
        "strategy":              strategy_name_from_path(file_path),
        "sample_count":          sample_count,
        "vm_count":              vm_count,
        "duration_mins":         duration_min,
        "total_cpu_joules":      totals["Energy_CPU_Joules"],
        "total_net_joules":      totals["Energy_Net_Joules"],
        "total_energy_grid_wh":  totals["Energy_Grid_Total_Wh"],
        "total_carbon_g":        totals["Carbon_Emissions_g"],
        "normalized_wh_per_min":     totals["Energy_Grid_Total_Wh"] / duration_min,
        "normalized_carbon_per_min": totals["Carbon_Emissions_g"]   / duration_min,
        "vm_summary": vm_summary.sort_values("Energy_Grid_Total_Wh", ascending=False),
    }


def print_strategy_summary(summary):
    print(f"=== STRATEGY: {summary['strategy']} ===")
    print(
        f"Valid Duration : {summary['duration_mins']:.1f} min | "
        f"Samples: {summary['sample_count']} | "
        f"VMs: {summary['vm_count']}\n"
        f"Raw Totals     → Grid: {summary['total_energy_grid_wh']:.4f} Wh | "
        f"Carbon: {summary['total_carbon_g']:.4f} g CO₂eq\n"
        f"Normalized     → {summary['normalized_wh_per_min']:.4f} Wh/min | "
        f"{summary['normalized_carbon_per_min']:.4f} g CO₂eq/min"
    )
    print("\nPer-VM energy totals (raw):")
    display = summary["vm_summary"].drop(columns=["Energy_Grid_Total_Joules"])
    print(display.round(4).to_string())
    print("-" * 60 + "\n")


def main():
    metrics_files = discover_metrics_files(METRICS_RESULTS_DIR)
    if not metrics_files:
        print(f"No strategy metrics files found in {METRICS_RESULTS_DIR}")
        return

    summaries = [summarize_strategy(f) for f in metrics_files]

    print("=== ENERGY AND CARBON SUMMARY PER STRATEGY ===\n")
    for s in summaries:
        print_strategy_summary(s)

    comparison = pd.DataFrame([
        {
            "Strategy":    s["strategy"],
            "Duration(m)": round(s["duration_mins"], 1),
            "Total_Wh":    s["total_energy_grid_wh"],
            "Total_CO2_g": s["total_carbon_g"],
            "Norm_Wh/min": s["normalized_wh_per_min"],
            "Norm_CO2/min": s["normalized_carbon_per_min"],
        }
        for s in summaries
    ]).sort_values("Norm_Wh/min")

    print("=== FAIR STRATEGY COMPARISON (sorted by normalized energy) ===")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()