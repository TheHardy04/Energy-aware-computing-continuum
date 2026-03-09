#!/bin/bash
set -e  # Exit on error

## GCP VM Startup Script for Apache Storm WORKER NODE
## This script runs automatically on VM creation/startup
## Logs are visible in: Compute Engine > VM instances > click VM > Logs > Serial port 1
## Or via: gcloud compute instances get-serial-port-output INSTANCE_NAME

# Log to both stdout and GCP serial console
exec > >(tee -a /var/log/storm-startup.log)
exec 2>&1

echo "=========================================="
echo "Storm WORKER Setup Started: $(date)"
echo "=========================================="

# Check if running as root (GCP startup scripts run as root by default)
if [[ $EUID -ne 0 ]]; then
   echo "Error: This script must be run as root" >&2
   exit 1
fi

install_worker_systemd_units() {
    echo "Configuring systemd units for worker services..."

    cat > /etc/systemd/system/storm-supervisor.service << 'EOF'
[Unit]
Description=Apache Storm Worker services (repo script)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=storm
Group=storm
Environment=STORM_HOME=/usr/local/storm
WorkingDirectory=/home/storm/Energy-aware-computing-continuum
ExecStart=/bin/bash -lc 'cd /home/storm/Energy-aware-computing-continuum && ./scripts/start_worker.sh'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable storm-supervisor.service
    systemctl restart storm-supervisor.service || true
    systemctl --no-pager --full status storm-supervisor.service || true
}

# Skip if already installed (for VM restarts)
if [[ -f "/usr/local/storm/bin/storm" ]]; then
    echo "Storm already installed, skipping setup"
    install_worker_systemd_units
    echo "Setup completed at: $(date)"
    echo "GCP_STARTUP_SCRIPT_STATUS: SUCCESS"
    exit 0
fi

# Update package list and install Java 17 and other necessary tools
echo "Installing prerequisites..."
# Use DEBIAN_FRONTEND=noninteractive for GCP automated environment
export DEBIAN_FRONTEND=noninteractive

# Fix any broken packages first
echo "Fixing package manager state..."
dpkg --configure -a 2>/dev/null || true
apt-get install -f -y -qq 2>/dev/null || true

# Update package list
apt-get update -qq || {
    echo "First apt-get update failed, trying again..."
    apt-get update
}

# Install packages with proper error handling
echo "Installing Java, Git, Maven, and other tools..."
apt-get update -qq || apt-get update  # Refresh right before install to avoid stale 404s
apt-get install -y -qq openjdk-17-jdk-headless wget curl tar git maven iputils-ping vim || {
    echo "Package installation encountered errors, attempting to fix..."
    apt-get update
    apt-get install -f -y
    apt-get install -y --fix-missing openjdk-17-jdk-headless wget curl tar git maven iputils-ping vim
}

# Zookeeper is not needed on worker nodes (only on master)

# Verify Java installation
java -version

# --- DOWNLOAD & INSTALL APACHE STORM 2.8.3 ---
echo "Installing Apache Storm..."
STORM_VER="${STORM_VER:-}"
if [[ -z "$STORM_VER" ]]; then
    echo "Resolving latest Apache Storm version..."
    STORM_MAJOR="${STORM_MAJOR:-2}"
    STORM_VER=$(curl -fsSL https://downloads.apache.org/storm/ \
        | grep -oE "apache-storm-${STORM_MAJOR}\.[0-9]+\.[0-9]+/" \
        | sed -E 's#apache-storm-([0-9]+\.[0-9]+\.[0-9]+)/#\1#' \
        | sort -V \
        | tail -n 1 || true)
fi

if [[ -z "$STORM_VER" ]]; then
    STORM_VER="2.8.3"
    echo "Warning: version discovery failed, falling back to $STORM_VER"
fi

echo "Using Apache Storm version: $STORM_VER"
STORM_TGZ="apache-storm-$STORM_VER.tar.gz"
STORM_URL="https://dlcdn.apache.org/storm/apache-storm-$STORM_VER/$STORM_TGZ"
cd /tmp
wget "$STORM_URL" || { echo "Error: Failed to download Apache Storm $STORM_VER from $STORM_URL." >&2; exit 1; }
tar -zxf "$STORM_TGZ"
mv apache-storm-$STORM_VER /usr/local/storm

# --- SET PERMISSIONS (SECURE) ---
# Create storm user and set proper permissions
useradd -r -m -s /bin/bash storm 2>/dev/null || true
mkdir -p /usr/local/storm/logs /usr/local/storm/data
chown -R storm:storm /usr/local/storm
chmod -R 755 /usr/local/storm

# --- GRANT STORM USER SUDO ACCESS ---
# Add storm to sudo group
usermod -aG sudo storm 2>/dev/null || true

# Create sudoers file for NOPASSWD access
cat > /etc/sudoers.d/storm-nopasswd << 'EOF'
# Allow storm user to run all commands without password
storm ALL=(ALL:ALL) NOPASSWD:ALL
EOF

chmod 0440 /etc/sudoers.d/storm-nopasswd

# Validate sudoers file
if visudo -c -f /etc/sudoers.d/storm-nopasswd; then
    echo "✓ Storm user granted sudo access without password"
else
    echo "⚠ Warning: sudoers file validation failed"
    rm -f /etc/sudoers.d/storm-nopasswd
fi

# Add binaries to PATH
ln -sf /usr/local/storm/bin/storm /usr/bin/storm

# Cleanup
rm -f "/tmp/$STORM_TGZ"

# Clone project to storm user's home first (needed for storm.yaml)
echo "Cloning project repository..."
if [ ! -d "/home/storm/Energy-aware-computing-continuum" ]; then
    sudo -u storm git clone https://github.com/TheHardy04/Energy-aware-computing-continuum.git /home/storm/Energy-aware-computing-continuum
    chown -R storm:storm /home/storm/Energy-aware-computing-continuum
    echo "✓ Project cloned to /home/storm/Energy-aware-computing-continuum"
else
    echo "✓ Project already exists"
fi

# Making all scripts in the project executable (in case permissions were lost during cloning)
find /home/storm/Energy-aware-computing-continuum -type f -name "*.sh" -exec chmod +x {} \;

# --- CONFIGURE STORM FOR WORKER NODE ---
echo "Configuring Storm for WORKER node..."

# Get nimbus IP from GCP metadata (REQUIRED for worker nodes)
NIMBUS_IP=$(curl -s -f -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/nimbus-ip" \
    2>/dev/null)

# Check if metadata fetch succeeded and returned a valid IP
if [[ -z "$NIMBUS_IP" ]] || [[ "$NIMBUS_IP" == *"<"* ]] || [[ "$NIMBUS_IP" == *"html"* ]]; then
    echo "Error: nimbus-ip metadata not found or invalid"
    echo "Worker nodes require the nimbus-ip metadata to be set"
    echo "Please redeploy with: --metadata nimbus-ip=<master-ip>"
    exit 1
fi

echo "Found nimbus-ip metadata: $NIMBUS_IP"
echo "Generating Storm configuration with Nimbus at $NIMBUS_IP"

# Generate storm.yaml dynamically with worker settings
cat > /usr/local/storm/conf/storm.yaml << STORM_CONFIG
storm.zookeeper.servers:
  - "$NIMBUS_IP"

nimbus.seeds: ["$NIMBUS_IP"]

# Ports of Storm supervisor
supervisor.slots.ports:
  - 6700

# custom scheduler class
storm.scheduler: "fr.dvrc.thardy.scheduler.CsvOneToOneScheduler"

# Path to the CSV file for the custom scheduler.
csv.scheduler.file: "/etc/storm/placement.csv"
csv.scheduler.hasHeader: true

# Rate that the topology emit a stat
topology.stats.sample.rate: 1

# Frequency that Nimbus give a sample
nimbus.monitor.freq.secs: 1

# Frequency that each task emit a sample
task.heartbeat.frequency.secs: 1

# Frequency that each executor emit a sample
executor.metrics.frequency.secs: 1
STORM_CONFIG

chown storm:storm /usr/local/storm/conf/storm.yaml
echo "✓ Storm configuration generated for WORKER"

# Create STORM_HOME environment variable for current shell, all users, and storm user shell
export STORM_HOME=/usr/local/storm
echo 'export STORM_HOME=/usr/local/storm' > /etc/profile.d/storm.sh
chmod 644 /etc/profile.d/storm.sh

# Setup .env file for storm user with STORM_HOME
echo "STORM_HOME=/usr/local/storm" > /home/storm/.env
if ! grep -q '^export STORM_HOME=/usr/local/storm$' /home/storm/.bashrc 2>/dev/null; then
    echo 'export STORM_HOME=/usr/local/storm' >> /home/storm/.bashrc
fi
chown storm:storm /home/storm/.env /home/storm/.bashrc

# Create placement CSV directory
mkdir -p /etc/storm
chmod 755 /etc/storm

echo "=========================================="
echo "✓ Storm WORKER setup complete!"
echo "  Version: $(storm version | head -1)"
echo "  Hostname: $(hostname -s)"
echo "  Nimbus IP: $NIMBUS_IP"
echo "  Time: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. SSH to this VM"
echo "  2. Run: systemctl status storm-supervisor"
echo ""

# Install and start systemd-managed worker service
install_worker_systemd_units

# Mark completion in GCP logs
echo "GCP_STARTUP_SCRIPT_STATUS: SUCCESS"
