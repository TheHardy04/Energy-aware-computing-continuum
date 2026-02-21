# Storm Scheduler Scripts

This directory contains shell scripts for managing Apache Storm clusters and submitting topologies.

**Important**: All scripts are designed to be executed from the `storm-scheduler` directory (parent directory), not from within the `scripts/` directory itself. The scripts will automatically change to the project root directory.

## Prerequisites

1. Apache Storm installed and `storm` command in PATH
2. ZooKeeper installed (optional, if `ZK_HOME` is set)
3. A `.env` file in the `storm-scheduler` directory with:
   ```bash
   STORM_HOME=/path/to/apache-storm
   ZK_HOME=/path/to/zookeeper  # Optional
   LOG_DIR=/path/to/logs       # Optional, defaults to storm-scheduler/logs
   ```

## Scripts Overview

### Environment Setup

#### `env.sh`
Sources the `.env` file and sets up environment variables for other scripts.
- Validates `STORM_HOME` is set
- Sets up derived paths (`STORM_BIN_DIR`, `STORM_CONF_DIR`, etc.)
- Creates log directory if needed

**Usage**: Sourced by other scripts (not run directly)

---

### Cluster Management

#### `start_master.sh`
Starts the Storm master node components (ZooKeeper, Nimbus, and UI).

**Usage from storm-scheduler directory**:
```bash
./scripts/start_master.sh
```

**What it does**:
1. Checks if ZooKeeper is running on port 2181, starts it if not
2. Starts Storm Nimbus
3. Starts Storm UI (accessible at http://localhost:8080)

---

#### `start_worker.sh`
Starts a single Storm supervisor node.

**Usage from storm-scheduler directory**:
```bash
./scripts/start_worker.sh
```

**What it does**:
- Starts a Storm supervisor using the default storm configuration
- Logs output to `logs/supervisor.log`

---

#### `deploy_local_supervisor_cluster.sh`
Starts multiple supervisor nodes using different configurations.

**Usage from storm-scheduler directory**:
```bash
./scripts/deploy_local_supervisor_cluster.sh
```

**What it does**:
- Reads all `conf/supervisor*/` directories
- Starts a separate supervisor for each configuration
- Each supervisor runs with its own `storm.yaml` config
- Logs are saved to `logs/supervisor1.log`, `logs/supervisor2.log`, etc.

**Example structure**:
```
conf/
  supervisor1/
    storm.yaml
  supervisor2/
    storm.yaml
  supervisor3/
    storm.yaml
```

---

#### `kill_storm.sh`
Stops all Storm processes (supervisors, nimbus, UI).

**Usage from storm-scheduler directory**:
```bash
./scripts/kill_storm.sh
```

**What it does**:
- Kills all Storm supervisor processes
- Kills Nimbus if running
- Kills Storm UI if running
- Uses multiple patterns to ensure all Storm Java processes are terminated
- Provides feedback on what was killed

**Fixed Issues**:
- Now uses more specific process patterns: `org.apache.storm.daemon.supervisor.Supervisor` and `org.apache.storm.daemon.nimbus.Nimbus`
- Changes to project root directory before running
- Better supervisor process detection

---

### Topology Submission

#### `test-topology-launch.sh`
Builds and submits the TestTopology (6-bolt synthetic benchmark).

**Usage from storm-scheduler directory**:
```bash
./scripts/test-topology-launch.sh
```

**What it does**:
1. Runs `mvn clean package` to build the project
2. Submits `TestTopology` to the Storm cluster
3. The topology runs a 6-bolt pipeline with CPU simulation

---

#### `launch-topology-from-csv.sh` ⭐ NEW
Builds and submits a topology generated from a properties file.

**Usage from storm-scheduler directory**:
```bash
./scripts/launch-topology-from-csv.sh <properties_file> [topology_name]
```

**Examples**:
```bash
# Submit 4-component application
./scripts/launch-topology-from-csv.sh ../python_algo/properties/Appli_4comps.properties MyApp

# Submit DCNS surveillance application (10 components)
./scripts/launch-topology-from-csv.sh ../python_algo/properties/Appli_10comps_dcns.properties DCNS

# Submit with default topology name
./scripts/launch-topology-from-csv.sh ../python_algo/properties/Appli_8comps_smartbuilding.properties
```

**What it does**:
1. Validates the properties file exists
2. Runs `mvn clean package` to build the project
3. Submits `TopologyFromCSV` with the specified properties file
4. Creates a Storm topology matching the application specification

See [TOPOLOGY_FROM_CSV.md](../TOPOLOGY_FROM_CSV.md) for more details.

---

### Other Scripts

#### `build_scheduler.sh`
Builds the project without submitting a topology.

**Usage from storm-scheduler directory**:
```bash
./scripts/build_scheduler.sh
```

---

#### `pull_newest_code.sh`
Git pull and build workflow (if using version control).

---

#### `vm_startup.sh`
Startup script for VM environments.

---

## Common Workflows

### Starting a Local Storm Cluster

```bash
cd /path/to/storm-scheduler

# Start master components (ZK, Nimbus, UI)
./scripts/start_master.sh

# Start worker nodes (choose one):
# Option 1: Single supervisor
./scripts/start_worker.sh

# Option 2: Multiple supervisors with different configs
./scripts/deploy_local_supervisor_cluster.sh

# Verify cluster is running
storm list
# Or visit http://localhost:8080
```

### Submitting a Topology

```bash
cd /path/to/storm-scheduler

# Option 1: Test topology (synthetic benchmark)
./scripts/test-topology-launch.sh

# Option 2: Topology from properties file
./scripts/launch-topology-from-csv.sh ../python_algo/properties/Appli_4comps.properties MyTopology
```

### Stopping Everything

```bash
cd /path/to/storm-scheduler

# Kill all running topologies (if any)
storm kill <topology_name>

# Stop all Storm processes
./scripts/kill_storm.sh

# Stop ZooKeeper (if you started it)
$ZK_HOME/bin/zkServer.sh stop
```

## Troubleshooting

### "storm: command not found"
Add Storm to your PATH:
```bash
export PATH=$PATH:/path/to/apache-storm/bin
```

### Logs location
Check logs at:
- `logs/nimbus.log` - Nimbus logs
- `logs/supervisor.log` or `logs/supervisor1.log` - Supervisor logs
- `logs/ui.log` - Storm UI logs

### Supervisors not stopping
The updated `kill_storm.sh` now uses more specific patterns:
- `org.apache.storm.daemon.supervisor.Supervisor` (primary pattern)
- `storm supervisor` (fallback pattern)

If processes still remain, manually kill them:
```bash
ps aux | grep storm
kill -9 <PID>
```

### Build errors
Ensure Maven is installed and Java 17+ is configured:
```bash
mvn --version
java --version
```

## Script Execution Directory

All scripts handle the execution directory properly:
1. They detect their own location using `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`
2. They calculate the project root as `PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"`
3. They change to the project root before executing Storm commands
4. This means you can run them from anywhere using `./scripts/script_name.sh`

**Recommended**: Always run from the `storm-scheduler` directory:
```bash
cd /path/to/storm-scheduler
./scripts/<script_name>.sh
```

