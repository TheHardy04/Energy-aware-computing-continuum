# Python Placement Framework

Placement and evaluation engine for service graphs over heterogeneous Cloud/Fog/Edge infrastructures.

## Install

```powershell
cd python_algo
pip install -r requirements.txt
```

## Basic Usage

```powershell
python main.py --strategy CSP
```

### Common examples

```powershell
python main.py --plot --verbose
python main.py --strategy GreedyFirstFit --infra properties/Infra_5nodes_GCP.properties --app properties/Appli_5comps_GCP.properties
python main.py --strategy LLM --infra properties/Infra_5nodes_GCP.properties --app properties/Appli_5comps_GCP.properties --placement-csv ..\results\placement.csv --metrics-csv ..\results\metrics_LLM.csv
```

## CLI Options

- `--strategy`: `CSP`, `LLM`, `GreedyFirstFit`, `GreedyFirstIterate`
- `--infra`: infrastructure properties file
- `--app`: application properties file
- `--placement-csv`: output placement CSV path
- `--metrics-csv`: output metrics CSV path
- `--plot`: draw infrastructure and service graphs
- `--verbose`: print graph, routing, and evaluation details

Default inputs:

- `properties/Infra_16nodes_fog3tier.properties`
- `properties/Appli_8comps_smartbuilding.properties`

## LLM Usage

The LLM strategy auto-detects a provider from the environment.

Supported providers:

- OpenAI via `OPENAI_API_KEY`
- Anthropic via `ANTHROPIC_API_KEY`
- Gemini via `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Ollama via local server on `OLLAMA_HOST` or default `http://localhost:11434`

Example with OpenAI:

```powershell
$env:OPENAI_API_KEY = "<your-key>"
$env:OPENAI_MODEL = "gpt-4o-mini"
python main.py --strategy LLM --infra properties/Infra_5nodes_GCP.properties --app properties/Appli_5comps_GCP.properties
```

Example with Ollama:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
python main.py --strategy LLM
```

## Energy Model Overrides

Use a different GCP energy properties file:

```powershell
$env:CSP_ENERGY_PROPERTIES = "D:\path\to\Energy_GCP.properties"
python main.py --strategy CSP
```

Optional solver energy tuning:

```powershell
$env:P_CPU_WATTS = "45"
$env:LLM_SOLVER_UNCERTAINTY = "3"
python main.py --strategy LLM
```

## Properties Files

- Infrastructure files live in [properties](properties)
- Application files live in [properties](properties)
- GCP energy constants live in [properties/Energy_GCP.properties](properties/Energy_GCP.properties)

Example property pair:

```powershell
python main.py --strategy CSP --infra properties/Infra_5nodes_GCP.properties --app properties/Appli_5comps_GCP.properties
```

## Outputs

- Placement CSV contains component-to-host mappings.
- Metrics CSV contains energy, latency, routing path, solver energy, and resource usage.

Example output paths:

```powershell
python main.py --strategy CSP --placement-csv ..\results\placement.csv --metrics-csv ..\results\metrics_CSP.csv
```

## Related Modules

- GCP deployment helpers: [../gcp_automations/README.md](../gcp_automations/README.md)
- Storm integration: [../storm-scheduler/README.md](../storm-scheduler/README.md)# Python Placement Framework

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
- `--placement-csv <path>`: export placement CSV.
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