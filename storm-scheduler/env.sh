#!/bin/bash

# --- env.sh ---
# This script sets up the environment variables for all other scripts.

# Load the .env file if it exists
if [ -f .env ]; then
    source .env
else
    echo "❌ Error: .env file not found."
    echo "   Please create one containing: STORM_HOME='/path/to/storm'"
    exit 1
fi

# Check if STORM_HOME is set
if [ -z "$STORM_HOME" ]; then
    echo "❌ Error: STORM_HOME is not set in .env"
    exit 1
fi

# Check if ZK_HOME is set (optional, only needed if zookeeper is not already running on the system)
if [ -z "$ZK_HOME" ]; then
    echo "⚠️  Warning: ZK_HOME is not set in .env."
    echo "   If you plan to start ZooKeeper from this project, please set ZK_HOME='/path/to/zookeeper' in your .env file."
fi

# Define derived paths (so other scripts don't have to)
export STORM_HOME
export STORM_BIN_DIR="$STORM_HOME/bin"
export STORM_LIB_DIR="$STORM_HOME/lib"
export STORM_CONF_DIR="$STORM_HOME/conf"
export LOG_DIR="logs"

# Verify that the bin directory exists
if [ ! -d "$STORM_BIN_DIR" ]; then
    echo "❌ Error: Storm bin directory not found at: $STORM_BIN_DIR"
    exit 1
fi

if [ -d "$STORM_CONF_DIR" ]; then
    echo "✅ Storm conf directory found at: $STORM_CONF_DIR"
    echo "  Copying storm.yaml to the conf directory"
    cp "conf/storm.yaml" "$STORM_CONF_DIR/storm.yaml"
else
    echo "⚠️  Warning: Storm conf directory not found at: $STORM_CONF_DIR"
    echo "   Please ensure Apache Storm is installed correctly and update the path in your .env file if needed."
fi

# Create a logs directory if it doesn't exist
mkdir -p "$LOG_DIR"
