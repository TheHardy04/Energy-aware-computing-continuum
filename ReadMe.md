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

## Full Experiment Automation

Run the full experiment with a single command:

```bash
./scripts/launch_placement_and_topology.sh ./configs/infra/Infra_5nodes_GCP.properties ./configs/app/Appli_5comps_GCP.properties ./configs/infra/Infra_5nodes_GCP_mapping.csv CSP
```

This workflow does the following in order:

1. Sources `scripts/env.sh` so Storm paths and logging variables are available.
2. Deploys or reuses the GCP infrastructure with `python gcp_automations/deploy_gcp_from_properties.py <properties_file>`.
3. Runs the Python placement engine and copies the generated CSVs into `/etc/storm/`.
4. Starts `services/python-placement/placement/src/realtime_telemetry.py` in the background and writes its logs to `experiments/raw/telemetry.log`.
5. Submits the Storm topology to Nimbus through `scripts/launch_topology_from_properties.sh`.

### Outputs

- Real-time metrics are appended to `experiments/raw/realtime_metrics.csv`.
- Telemetry logs are written to `experiments/raw/telemetry.log`.
- The telemetry PID is stored in `/tmp/telemetry.pid` so it can be stopped by `scripts/kill_storm.sh`.

### Energy Mapping

The real-time daemon mirrors the theoretical placement model used by the algorithms:

- Compute power is derived from the same GCP vCPU-slot energy model used by placement evaluation.
- Network power uses the measured GCP `sent_bytes_count` metric over a 30-second window and the constant `E_BIT = 1e-7` J/bit.
- Total power is reported as `Compute_Power_W + Network_Power_W`, which makes the live telemetry directly comparable with the theoretical placement objective.

In practice, this means your offline placement energy and the live experiment telemetry are expressed with the same compute and network terms, so you can compare algorithmic predictions against measured runtime behavior without changing the downstream analysis pipeline.

## Documentation

- [services/python-placement/README.md](services/python-placement/README.md)
- [gcp_automations/README.md](gcp_automations/README.md)
- [services/java-storm-scheduler/README.md](services/java-storm-scheduler/README.md)
- [scripts/README.md](scripts/README.md)
- [configs/README.md](configs/README.md)
- [experiments/README.md](experiments/README.md)