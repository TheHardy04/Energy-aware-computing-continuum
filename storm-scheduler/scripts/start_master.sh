#!/bin/bash

## This script starts the master node components of Apache Storm: ZooKeeper, Nimbus, and the Storm UI.

# --- CONFIGURATION ---

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables
source "$SCRIPT_DIR/env.sh"

mkdir -p "$LOG_DIR"

echo "=============== Starting Master Node Components... ==============="

# 1. START ZOOKEEPER IF NEEDED
if nc -z localhost 2181; then
  echo "   > ZooKeeper is already running on port 2181."
else
    echo "   > Starting ZooKeeper..."
    $ZK_HOME/bin/zkServer.sh start
fi

echo "     ✅ ZooKeeper is UP."

# 2. START NIMBUS
echo "   > Starting Nimbus..."
if nc -z localhost 6627; then
  echo "   > Nimbus is already running on port 6627."
else
    nohup $STORM_HOME/bin/storm nimbus > "$LOG_DIR/nimbus.log" 2>&1 &
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

echo "-----------------------------------------------"
echo " Master Node is Ready!"
echo "   - Dashboard: http://localhost:8080"
echo "   - Now you can launch your Supervisors."
echo "-----------------------------------------------"