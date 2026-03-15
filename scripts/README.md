# Scripts

Helper scripts for local Storm setup, scheduler deployment, topology submission, and cluster cleanup.

## Prerequisites

Create a `.env` file in the repository root or your home directory.

Minimum variables:

```bash
STORM_HOME=/path/to/apache-storm-2.8.3
```

Optional variables:

```bash
ZK_HOME=/path/to/zookeeper
LOG_DIR=/tmp/storm-logs
```

## Common Commands

### Build and deploy the scheduler JAR into Storm

```bash
./scripts/build_scheduler.sh
```

### Start the master services

Starts ZooKeeper if needed, then Nimbus and the Storm UI.

```bash
./scripts/start_master.sh
```

### Start a worker supervisor

```bash
./scripts/start_worker.sh
```

### Submit a topology from an application properties file

```bash
./scripts/launch_topology_from_properties.sh ./python_algo/properties/Appli_4comps.properties DemoTopology
```

### Run placement and submit the topology in one step

This wrapper runs the Python placement, writes outputs to `results/`, copies the scheduler CSV files into `/etc/storm/`, and then submits the topology.

```bash
./scripts/launch_placement_and_topology.sh ./python_algo/properties/Infra_5nodes_GCP.properties ./python_algo/properties/Appli_5comps_GCP.properties ./python_algo/properties/Infra_5nodes_GCP_mapping.csv CSP
```

### Submit the Java test topology

```bash
./scripts/test-topology-launch.sh
```

### Stop all Storm processes

```bash
./scripts/kill_storm.sh
```

## Debug and Local Helpers

Launch multiple local supervisors from `storm-scheduler/conf/supervisor*/`:

```bash
./scripts/deploy_local_supervisor_cluster.sh
```

Inspect supervisor identity and placement CSV resolution:

```bash
./scripts/debug_vm_identification.sh
```

## Placement + Storm Workflow

A typical manual flow is:

```bash
python ./python_algo/main.py --strategy CSP --infra ./python_algo/properties/Infra_5nodes_GCP.properties --app ./python_algo/properties/Appli_5comps_GCP.properties --placement-csv ./results/placement.csv --metrics-csv ./results/metrics_CSP.csv
./scripts/build_scheduler.sh
./scripts/start_master.sh
./scripts/start_worker.sh
./scripts/launch_topology_from_properties.sh ./python_algo/properties/Appli_5comps_GCP.properties DemoTopology
```

## Notes

- `env.sh` is sourced by the other scripts and validates `STORM_HOME`.
- `launch_placement_and_topology.sh` writes placement and metrics outputs to [../results](../results).
- The combined wrapper expects the Storm config created by the GCP startup scripts, where `csv.scheduler.file` points to `/etc/storm/placement.csv`.