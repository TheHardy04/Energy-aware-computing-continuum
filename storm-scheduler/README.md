# Storm Scheduler

Apache Storm integration layer for the placement framework. It contains custom schedulers plus Java entrypoints that build a topology from application properties files.

## Requirements

- Java 17+
- Maven 3.9+
- Apache Storm 2.8.3

## Build

```bash
cd storm-scheduler
mvn clean package
```

Artifacts:

- `target/storm-scheduler-1.0-SNAPSHOT.jar`
- `target/storm-scheduler-1.0-SNAPSHOT-all.jar`

## Run a Topology From Properties

From the repository root:

```bash
./scripts/launch_topology_from_properties.sh ./python_algo/properties/Appli_4comps.properties DemoTopology
./scripts/launch_topology_from_properties.sh ./python_algo/properties/Appli_10comps_dcns.properties DCNS
```

This script:

- builds the Maven module
- checks that the `storm` CLI is available
- submits `fr.dvrc.thardy.topology.TopologyFromProperties`

## Submit the Test Topology

```bash
./scripts/test-topology-launch.sh
```

## Components

- `fr.dvrc.thardy.topology.TopologyFromProperties`: creates a topology from an app properties file
- `fr.dvrc.thardy.topology.TestTopology`: synthetic test topology
- `fr.dvrc.thardy.scheduler.CsvOneToOneScheduler`: placement-aware scheduler using CSV inputs
- `fr.dvrc.thardy.scheduler.DefaultSchedulerRework`: custom scheduler variant built on Storm defaults

## Cluster Notes

- The scheduler module is designed to work with the helper scripts in [../scripts/README.md](../scripts/README.md).
- For GCP deployments, worker bootstrap is handled by [../gcp_automations/vm_startup.sh](../gcp_automations/vm_startup.sh).

## Storm UI Tunnel

```bash
gcloud compute ssh storm-nimbus --project=<PROJECT_ID> --zone=<ZONE> -- -L 8080:localhost:8080
```

Then open `http://localhost:8080`.

## Troubleshooting

If log files are not writable, set a writable log directory in your `.env` file:

```bash
LOG_DIR=/tmp/storm-logs
```

If the `storm` command is missing, add `$STORM_HOME/bin` to `PATH` before using the helper scripts.
