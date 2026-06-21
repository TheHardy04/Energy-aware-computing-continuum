# Energy-aware Computing Continuum

This repository studies energy-aware service placement across the cloud, fog, and edge continuum. It combines a Python placement engine, GCP automation for VM provisioning and monitoring, and an Apache Storm scheduler layer for topology execution.

## Repository Layout

- `configs/`: versioned experiment inputs and shared model settings.
	- `configs/infra/`: infrastructure `.properties` files and VM mapping CSVs.
	- `configs/app/`: application `.properties` files.
	- `configs/energy/`: shared GCP energy model settings.
- `services/python-placement/`: Python placement, evaluation, and CSV export code.
- `gcp_automations/`: scripts that deploy GCP VMs and collect Monitoring metrics.
- `services/java-storm-scheduler/`: Apache Storm topology builder and custom schedulers.
- `scripts/`: orchestration scripts for local Storm and end-to-end runs.
- `experiments/`: generated results, benchmark CSVs, and analysis outputs.

## Typical Workflow

1. Choose an infrastructure file from `configs/infra/` and an application file from `configs/app/`.
2. Run a placement with `services/python-placement/placement/main.py`.
3. Write placement and metrics outputs under `experiments/results/`.
4. Deploy matching GCP VMs with `gcp_automations/deploy_gcp_from_properties.py` when needed.
5. Build and submit the Storm topology with the scripts in `scripts/`.

## Quick Start

```bash
python services/python-placement/placement/main.py --strategy CSP --infra configs/infra/Infra_5nodes_GCP.properties --app configs/app/Appli_5comps_GCP.properties --placement-csv experiments/results/placement.csv --metrics-csv experiments/results/metrics_CSP.csv
```

```bash
python gcp_automations/deploy_gcp_from_properties.py configs/infra/Infra_5nodes_GCP.properties
```

```bash
./scripts/launch_topology_from_properties.sh configs/app/Appli_4comps.properties DemoTopology
```

```bash
./scripts/launch_placement_and_topology.sh configs/infra/Infra_5nodes_GCP.properties configs/app/Appli_5comps_GCP.properties configs/infra/Infra_5nodes_GCP_mapping.csv CSP
```

## Documentation

- [services/python-placement/README.md](services/python-placement/README.md)
- [gcp_automations/README.md](gcp_automations/README.md)
- [services/java-storm-scheduler/README.md](services/java-storm-scheduler/README.md)
- [scripts/README.md](scripts/README.md)
- [configs/README.md](configs/README.md)
- [experiments/README.md](experiments/README.md)