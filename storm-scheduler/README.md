# Storm Scheduler

This project implements energy-aware scheduling for Apache Storm in fog/edge/cloud computing continuum environments.

## 🚀 Quick Start

### Submit a Topology from Properties File (NEW!)

```bash
# Build and submit a 4-component application
./scripts/launch-topology-from-csv.sh ../python_algo/properties/Appli_4comps.properties MyApp

# Submit a DCNS surveillance application (10 components)
./scripts/launch-topology-from-csv.sh ../python_algo/properties/Appli_10comps_dcns.properties DCNS
```

See [TOPOLOGY_FROM_CSV.md](TOPOLOGY_FROM_CSV.md) for complete documentation on the TopologyFromCSV feature.

### Submit Test Topology

```bash
./scripts/test-topology-launch.sh
```

## Features

1. **TopologyFromCSV** ⭐ - Generate Storm topologies from application properties files
2. **TestTopology** - Synthetic 6-bolt benchmark topology
3. **Custom Schedulers** - Energy-aware scheduling algorithms
4. **Shell Scripts** - Convenient cluster management (see [scripts/README.md](scripts/README.md))

---

## Installation

To run the Storm Scheduler, you need to have Apache Storm installed and set up. You can find the installation instructions on the official [Apache Storm website](https://storm.apache.org/index.html).

Version `2.8.3` was used for testing.
Ubuntu `22.04` and `24.04` was used for testing.

## Apache Storm Installation Steps

```shell
# Update package list and install Java 17 
apt-get update
apt-get install -y openjdk-17-jdk-headless wget python3 tar

# Verify Java installation
java -version

# --- 2. INSTALL ZOOKEEPER ---
apt-get install -y zookeeperd

# --- 3. DOWNLOAD & INSTALL APACHE STORM 2.8.3 ---
STORM_VER="2.8.3"
wget https://downloads.apache.org/storm/apache-storm-$STORM_VER/apache-storm-$STORM_VER.tar.gz
tar -zxf apache-storm-$STORM_VER.tar.gz
mv apache-storm-$STORM_VER /usr/local/storm

# Add binaries to PATH
export PATH=$PATH:/usr/local/storm/bin
```

## GCP

### Create a GCP VM

When creating a VM in GCP, make sure to allow HTTP and HTTPS traffic. You can do this by selecting the appropriate checkboxes in the "Firewall" section of the VM creation form.

You can use the `vm_startup.sh` script to automate the installation of Apache Storm on the VM. 

### To see UI

On master node, run the following command to forward the port for Storm UI:

```shell
gcloud compute ssh storm-nimbus \
    --project=<PROJECT_ID> \
    --zone=<ZONE> \
    -- -L 8080:localhost:8080
```

## Troubleshooting

### Permission Denied Error for Logs

If you encounter an error like:
```
Permission denied: /path/to/logs/supervisor.log
```

**Solution 1: Manually fix permissions**
```bash
# Create logs directory if it doesn't exist
mkdir -p logs

# Give yourself write permissions
chmod -R u+rwx logs

# Or, if needed, take ownership
sudo chown -R $USER logs
```

**Solution 2: Use a different log directory**
Edit the `.env` file and add:
```
LOG_DIR=/tmp/storm-logs
```
