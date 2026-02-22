#!/bin/bash
set -e  # Exit on error

## GCP VM Startup Script for Apache Storm
## This script runs automatically on VM creation/startup
## Logs are visible in: Compute Engine > VM instances > click VM > Logs > Serial port 1
## Or via: gcloud compute instances get-serial-port-output INSTANCE_NAME

# Log to both stdout and GCP serial console
exec > >(tee -a /var/log/storm-startup.log)
exec 2>&1

echo "=========================================="
echo "Storm Setup Started: $(date)"
echo "=========================================="

# Check if running as root (GCP startup scripts run as root by default)
if [[ $EUID -ne 0 ]]; then
   echo "Error: This script must be run as root" >&2
   exit 1
fi

# Skip if already installed (for VM restarts)
if [[ -f "/usr/local/storm/bin/storm" ]]; then
    echo "Storm already installed, skipping setup"
    echo "Setup completed at: $(date)"
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
apt-get install -y -qq openjdk-17-jdk-headless wget python3 tar git maven python3-pip vim || {
    echo "Package installation encountered errors, attempting to fix..."
    apt-get install -f -y
    apt-get install -y openjdk-17-jdk-headless wget python3 tar git maven python3-pip python3-full python3-venv vim
}

# Install zookeeper separately (can fail on some systems)
echo "Installing Zookeeper..."
apt-get install -y -qq zookeeperd || {
    echo "⚠ Warning: Zookeeper installation failed. Install manually if needed."
    echo "  This is only required on master nodes."
}

# Verify Java installation
java -version

# --- DOWNLOAD & INSTALL APACHE STORM 2.8.3 ---
echo "Installing Apache Storm..."
STORM_VER="2.8.3"
cd /tmp
wget -q https://downloads.apache.org/storm/apache-storm-$STORM_VER/apache-storm-$STORM_VER.tar.gz || { echo "Error: Failed to download Apache Storm $STORM_VER." >&2; exit 1; }
tar -zxf apache-storm-$STORM_VER.tar.gz
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
rm -f /tmp/apache-storm-$STORM_VER.tar.gz

# Clone project to storm user's home first (needed for storm.yaml)
echo "Cloning project repository..."
if [ ! -d "/home/storm/Energy-aware-computing-continuum" ]; then
    sudo -u storm git clone https://github.com/TheHardy04/Energy-aware-computing-continuum.git /home/storm/Energy-aware-computing-continuum
    chown -R storm:storm /home/storm/Energy-aware-computing-continuum
    echo "✓ Project cloned to /home/storm/Energy-aware-computing-continuum"
else
    echo "✓ Project already exists"
fi

# --- CONFIGURE STORM ---
echo "Configuring Storm..."

# Copy storm.yaml from repo
if [ -f "/home/storm/Energy-aware-computing-continuum/storm-scheduler/conf/storm.yaml" ]; then
    cp /home/storm/Energy-aware-computing-continuum/storm-scheduler/conf/storm.yaml /usr/local/storm/conf/storm.yaml
    chown storm:storm /usr/local/storm/conf/storm.yaml
    echo "✓ Storm configured (copied from repo)"
else
    echo "⚠ Warning: storm.yaml not found in repo at storm-scheduler/conf/storm.yaml"
    echo "  Storm will use default configuration"
fi

# Setup .env file for storm user with STORM_HOME
echo "STORM_HOME=/usr/local/storm" > /home/storm/.env
chown storm:storm /home/storm/.env

# Setup python virtual environment for storm user
sudo -u storm python3 -m venv /home/storm/venv
chown -R storm:storm /home/storm/venv
source /home/storm/venv/bin/activate
pip install --upgrade pip || echo "⚠ Warning: pip upgrade failed, continuing with existing pip version"
pip install -r /home/storm/Energy-aware-computing-continuum/python_algo/requirements.txt || echo "⚠ Warning: Failed to install Python dependencies, please check the requirements.txt file and your network connection."

# Create placement CSV directory
mkdir -p /etc/storm
chmod 755 /etc/storm

echo "=========================================="
echo "✓ Storm setup complete!"
echo "  Version: $(storm version | head -1)"
echo "  Hostname: $(hostname -s)"
echo "  Time: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. SSH to this VM"
echo "  2. Run: cd /home/storm/Energy-aware-computing-continuum/storm-scheduler"
echo "  3. Master: ./scripts/start_master.sh"
echo "     Worker: ./scripts/start_worker.sh"
echo ""

# Mark completion in GCP logs
echo "GCP_STARTUP_SCRIPT_STATUS: SUCCESS"
