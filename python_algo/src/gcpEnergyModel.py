"""
GCP Energy Model - Shared utilities for GCP vCPU-slot energy model.

This module provides functions to load and compute energy metrics using the GCP
energy model with latency-tiered network factors and PUE-aware vCPU power consumption.

Constants:
    GCP_PUE: Google Cloud Platform Power Usage Effectiveness
    GCP_P_VCPU_IDLE_W: Idle power per vCPU (Watts)
    GCP_P_VCPU_ACTIVE_W: Active power per vCPU (Watts)
    GCP_LAT_INTRAZONE_MS: Intra-zone latency threshold (ms)
    GCP_LAT_INTERZONE_MS: Inter-zone latency threshold (ms)
    GCP_FACTOR_INTRAZONE: Network energy factor for intra-zone links
    GCP_FACTOR_INTERZONE: Network energy factor for inter-zone links
    GCP_FACTOR_CROSSREGION: Network energy factor for cross-region links
    ENERGY_SCALE: Integer scaling factor for CP-SAT (CSP) models
"""

from typing import Dict
import os
from functools import lru_cache

# GCP energy model defaults (aligned with Google Environmental Report 2023)
GCP_PUE = 1.10
GCP_P_VCPU_IDLE_W = 1.0
GCP_P_VCPU_ACTIVE_W = 8.0

GCP_LAT_INTRAZONE_MS = 2.0
GCP_LAT_INTERZONE_MS = 25.0

GCP_FACTOR_INTRAZONE = 1.0
GCP_FACTOR_INTERZONE = 1.5
GCP_FACTOR_CROSSREGION = 2.0

# CP-SAT works on integers; energies are tracked in deci-Watts and converted back to Watts in meta.
ENERGY_SCALE = 10


def _parse_simple_properties(file_path: str) -> Dict[str, str]:
    """Minimal .properties parser (key=value), compatible with project files.
    
    Args:
        file_path: Path to the .properties file
        
    Returns:
        Dictionary mapping property keys to values
    """
    props: Dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        continuation = ""
        for raw in f:
            line = raw.rstrip("\n")
            if line.endswith("\\"):
                continuation += line[:-1]
                continue
            line = (continuation + line).strip()
            continuation = ""
            if not line or line[0] in "#!":
                continue
            if "=" in line:
                k, v = line.split("=", 1)
            elif ":" in line:
                k, v = line.split(":", 1)
            else:
                continue
            props[k.strip()] = v.strip()
    return props


@lru_cache(maxsize=1)
def _load_energy_settings(override_path: str = "") -> Dict[str, float]:
    """Load GCP energy constants from Energy_GCP.properties with safe defaults.
    
    Args:
        override_path: Optional explicit path to properties file. If not provided,
                      checks CSP_ENERGY_PROPERTIES env var, then default location.
                      
    Returns:
        Dictionary with GCP energy model configuration values
    """
    default_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "properties", "Energy_GCP.properties")
    )
    config_path = (
        override_path
        or os.environ.get("CSP_ENERGY_PROPERTIES", "")
        or default_path
    )

    settings = {
        "gcp.pue": GCP_PUE,
        "gcp.p_vcpu_idle_w": GCP_P_VCPU_IDLE_W,
        "gcp.p_vcpu_active_w": GCP_P_VCPU_ACTIVE_W,
        "gcp.lat.intrazone_ms": GCP_LAT_INTRAZONE_MS,
        "gcp.lat.interzone_ms": GCP_LAT_INTERZONE_MS,
        "gcp.factor.intrazone": GCP_FACTOR_INTRAZONE,
        "gcp.factor.interzone": GCP_FACTOR_INTERZONE,
        "gcp.factor.crossregion": GCP_FACTOR_CROSSREGION,
        "gcp.energy_scale": float(ENERGY_SCALE),
    }

    if os.path.exists(config_path):
        try:
            raw = _parse_simple_properties(config_path)
            for key in list(settings.keys()):
                if key in raw:
                    settings[key] = float(raw[key])
        except Exception:
            # Keep defaults if the settings file is malformed.
            pass

    # Return a copy to prevent cached dict mutation
    return settings.copy()


def link_factor(latency_ms: float, cfg: Dict[str, float]) -> float:
    """Infer GCP network tier factor from measured latency.
    
    Args:
        latency_ms: Link latency in milliseconds
        cfg: Configuration dictionary from _load_energy_settings()
        
    Returns:
        Energy factor (1.0 for intra-zone, 1.5 for inter-zone, 2.0 for cross-region)
    """
    if latency_ms <= cfg["gcp.lat.intrazone_ms"]:
        return cfg["gcp.factor.intrazone"]
    if latency_ms <= cfg["gcp.lat.interzone_ms"]:
        return cfg["gcp.factor.interzone"]
    return cfg["gcp.factor.crossregion"]


def link_factor_scaled(latency_ms: float, cfg: Dict[str, float]) -> int:
    """Return GCP link factor scaled by ENERGY_SCALE for integer CP-SAT models.
    
    Args:
        latency_ms: Link latency in milliseconds
        cfg: Configuration dictionary from _load_energy_settings()
        
    Returns:
        Scaled integer factor (10, 15, or 20 with default ENERGY_SCALE=10)
    """
    intrazone_ms = cfg["gcp.lat.intrazone_ms"]
    interzone_ms = cfg["gcp.lat.interzone_ms"]
    energy_scale = int(round(cfg["gcp.energy_scale"]))

    if latency_ms <= intrazone_ms:
        return int(round(cfg["gcp.factor.intrazone"] * energy_scale))
    if latency_ms <= interzone_ms:
        return int(round(cfg["gcp.factor.interzone"] * energy_scale))
    return int(round(cfg["gcp.factor.crossregion"] * energy_scale))


def node_power_w(cpu_used: int, cpu_cap: int, cfg: Dict[str, float]) -> float:
    """GCP vCPU-slot node power model with explicit zero for inactive VMs.
    
    Implements the linear power model:
        P = PUE × [P_idle × cpu_cap + (P_active - P_idle) × cpu_used]
    
    Args:
        cpu_used: Number of vCPUs currently used
        cpu_cap: Total vCPU capacity of the host
        cfg: Configuration dictionary from _load_energy_settings()
        
    Returns:
        Power in Watts (0.0 if VM is inactive)
    """
    if cpu_cap <= 0 or cpu_used <= 0:
        return 0.0
    used = min(cpu_used, cpu_cap)
    return cfg["gcp.pue"] * (
        cfg["gcp.p_vcpu_idle_w"] * cpu_cap
        + (cfg["gcp.p_vcpu_active_w"] - cfg["gcp.p_vcpu_idle_w"]) * used
    )
