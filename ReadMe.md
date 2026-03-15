# Energy-aware Computing Continuum

Energy-aware service placement for Cloud, Fog, and Edge infrastructures. The repository contains a Python placement framework, GCP deployment helpers, and an Apache Storm integration layer.

## Modules

- [python_algo](python_algo): placement algorithms, evaluators, properties files, and CSV exports.
- [gcp_automations](gcp_automations): GCP VM provisioning and Monitoring data collection.
- [storm-scheduler](storm-scheduler): Apache Storm topologies and custom schedulers.
- [scripts](scripts): helper scripts to build, start, stop, and submit Storm jobs.

## Prerequisites

- Python 3.12+
- Java 17+
- Maven 3.9+
- Apache Storm 2.8.3 for cluster execution
- Google Cloud SDK for GCP deployment

## Quick Start

### Run a placement locally

```powershell
cd python_algo
pip install -r requirements.txt
python main.py --strategy CSP --verbose
```

### Run a placement with explicit properties

```powershell
cd python_algo
python main.py --strategy GreedyFirstFit --infra properties/Infra_5nodes_GCP.properties --app properties/Appli_5comps_GCP.properties --placement-csv ..\results\placement.csv --metrics-csv ..\results\metrics_GreedyFirstFit.csv
```

### Deploy a GCP infrastructure from an infra properties file

```powershell
pip install -r gcp_automations/requirements.txt
python gcp_automations/deploy_gcp_from_properties.py python_algo/properties/Infra_5nodes_GCP.properties
```

### Submit a Storm topology from an application properties file

```bash
./scripts/launch_topology_from_properties.sh ./python_algo/properties/Appli_4comps.properties DemoTopology
```

### Run placement and topology submission together

```bash
./scripts/launch_placement_and_topology.sh ./python_algo/properties/Infra_5nodes_GCP.properties ./python_algo/properties/Appli_5comps_GCP.properties ./python_algo/properties/Infra_5nodes_GCP_mapping.csv CSP
```

## Typical Workflow

1. Define infrastructure and application inputs in [python_algo/properties](python_algo/properties).
2. Run a placement strategy with [python_algo/main.py](python_algo/main.py).
3. Export placement and metrics CSV files to [results](results) or [results_infra10](results_infra10).
4. If needed, provision matching GCP VMs with [gcp_automations/deploy_gcp_from_properties.py](gcp_automations/deploy_gcp_from_properties.py).
5. Build and submit the Storm topology with the scripts in [scripts](scripts).

## Outputs

- Placement mapping CSV: `results/placement.csv` unless overridden.
- Metrics CSV: appended to the path passed with `--metrics-csv`.
- Example benchmark outputs already exist in [results](results) and [results_infra10](results_infra10).

## Module Docs

- [python_algo/ReadMe.md](python_algo/ReadMe.md)
- [gcp_automations/README.md](gcp_automations/README.md)
- [storm-scheduler/README.md](storm-scheduler/README.md)
- [scripts/README.md](scripts/README.md)