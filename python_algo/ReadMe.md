# Python Placement Framework

## Prerequisites

- Python 3.12+
- Dependencies from `requirements.txt`

## Installation

```bash
cd python_algo
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## CLI Arguments

- `--plot`: plot infrastructure and service graphs.
- `--verbose`: print detailed graph, placement, and evaluation output.
- `--infra <path>`: infrastructure properties file.
  - default: `properties/Infra_16nodes_fog3tier.properties`
- `--app <path>`: application properties file.
  - default: `properties/Appli_8comps_smartbuilding.properties`
- `--strategy <strategy>`: one of:
  - `CSP`
  - `LLM`
  - `GreedyFirstFit`
  - `GreedyFirstIterate`
- `--to-csv <path>`: export placement CSV.
  - default: `results/placement.csv`

## Examples

```bash
python main.py --plot --verbose
python main.py --strategy LLM --verbose
python main.py --strategy CSP --infra properties/Infra_16nodes_fog3tier.properties --app properties/Appli_10comps_dcns.properties
```

## GCP Energy Settings

Energy-related metrics in `src/cspPlacement.py`, `src/llmPlacement.py`, and `src/evaluation.py` use a shared GCP settings file:

- `properties/Energy_GCP.properties`

### Supported keys

- `gcp.pue`
- `gcp.p_vcpu_idle_w`
- `gcp.p_vcpu_active_w`
- `gcp.lat.intrazone_ms`
- `gcp.lat.interzone_ms`
- `gcp.factor.intrazone`
- `gcp.factor.interzone`
- `gcp.factor.crossregion`
- `gcp.energy_scale` (used by CSP integer scaling)

### Override file path

Set an alternate settings file with:

```bash
# PowerShell
$env:CSP_ENERGY_PROPERTIES = "D:\path\to\Energy_GCP.properties"
python main.py --strategy CSP
```

## How Link Types Are Specified

Link types are inferred from edge latency in `network.topology`:

- latency `<= gcp.lat.intrazone_ms` -> intra-zone
- latency `<= gcp.lat.interzone_ms` -> inter-zone
- latency `> gcp.lat.interzone_ms` -> cross-region

In infra properties, links are still declared as:

```properties
network.topology={src,dst,bandwidth,latency},...
```

## Features

- Infrastructure graph + service graph modeling
- Placement strategies: CSP, LLM, GreedyFirstFit, GreedyFirstIterate
- Constraint checks (DZ, CPU/RAM, connectivity, bandwidth)
- Evaluation with GCP-aligned energy metrics
- CSV export and optional plotting

## GCP Integration

The infrastructure properties files in `python_algo/properties/` can be used directly by:

- `gcp_automations/deploy_gcp_from_properties.py`

Example:

```bash
python .\gcp_automations\deploy_gcp_from_properties.py .\python_algo\properties\Infra_5nodes_GCP.properties
```

Notes:

- VM sizing is derived from `hosts.configuration` tuples (`{cpu, ram}`).
- Worker provisioning is now Java/Storm-focused for faster startup (no Python virtual environment installation on workers).