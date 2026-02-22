#!/bin/bash

## This script starts the master node components of Apache Storm: ZooKeeper, Nimbus, and the Storm UI.

# --- CONFIGURATION ---

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# Load environment variables
source "$SCRIPT_DIR/env.sh"

# Create log directory if it doesn't exist
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    echo "❌ Error: Cannot create log directory at $LOG_DIR"
    echo "   Please check permissions or set LOG_DIR to a writable location in env.sh"
    exit 1
fi

# Verify we can write to the log directory
if [ ! -w "$LOG_DIR" ]; then
    echo "❌ Error: No write permission for log directory: $LOG_DIR"
    echo "   Please run: chmod u+w '$LOG_DIR' or change LOG_DIR in env.sh"
    exit 1
fi


# Check if netcat (nc) is installed for port checking
if ! command -v nc &> /dev/null; then
    echo "Error: 'nc' (netcat) is required but not installed."
    echo "Trying to install netcat..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y netcat-openbsd
    fi
fi

if ! command -v nc &> /dev/null; then
    echo "Error: 'nc' (netcat) is still not available. Please install it manually and re-run this script."
    exit 1
fi

echo "=============== Starting Master Node Components... ==============="

# 1. START ZOOKEEPER IF NEEDED
if nc -z localhost 2181; then
  echo "   > ZooKeeper is already running on port 2181."
else
    echo "   > Starting ZooKeeper..."
    $ZK_HOME/bin/zkServer.sh start
fi

echo "     ✅ ZooKeeper is UP."

# 2. Build Scheduler JAR (if needed)
source "$SCRIPT_DIR/build_scheduler.sh"

# 2. START NIMBUS
echo "   > Starting Nimbus..."
if nc -z localhost 6627; then
  echo "   > Nimbus is already running on port 6627."
else
    nohup $STORM_HOME/bin/storm nimbus > "$LOG_DIR/nimbus.log" 2>&1 &

    NIMBUS_PID=$!
    # Wait a moment to allow Nimbus to start and write logs
    sleep 5
    # Check if the Nimbus process is still running
    if ps -p $NIMBUS_PID > /dev/null; then
        echo "Nimbus started successfully (PID: $NIMBUS_PID)"
    else
        echo "Failed to start Nimbus. Check $LOG_DIR/nimbus.log for details"
        exit 1
    fi
fi
echo "     ✅ Nimbus launched (PID: $!). Logs in $LOG_DIR/nimbus.log"

# 3. START UI
echo "   > Starting Storm UI..."
if nc -z localhost 8080; then
    echo "   > Something is already running on port 8080. Please check if the Storm UI is already running or if another service is using that port."
else
    nohup $STORM_HOME/bin/storm ui > "$LOG_DIR/ui.log" 2>&1 &
    echo "     ✅ Storm UI launched (PID: $!). Logs in $LOG_DIR/ui.log"
fi

# get public IP address for user reference
PUBLIC_IP=$(curl -s ifconfig.me || echo "localhost")

echo "-----------------------------------------------"
echo " Master Node is Ready!"
echo "   - Dashboard: http://$PUBLIC_IP:8080"
echo "   - Now you can launch your Supervisors."
echo "-----------------------------------------------"