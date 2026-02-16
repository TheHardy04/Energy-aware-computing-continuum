#!/bin/bash

# --- env.sh ---
# This script sets up the environment variables for all other scripts.

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load the .env file if it exists (from project root)
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
else
    echo "❌ Error: .env file not found in $PROJECT_ROOT"
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
export PROJECT_ROOT
export LOG_DIR="$PROJECT_ROOT/logs"

# Verify that the bin directory exists
if [ ! -d "$STORM_BIN_DIR" ]; then
    echo "❌ Error: Storm bin directory not found at: $STORM_BIN_DIR"
    exit 1
fi

if [ -d "$STORM_CONF_DIR" ]; then
    echo "✅ Storm conf directory found at: $STORM_CONF_DIR"
    echo "  Copying storm.yaml to the conf directory"
    if ! cp "$PROJECT_ROOT/conf/storm.yaml" "$STORM_CONF_DIR/storm.yaml"; then
        echo "❌ Error: Failed to copy $PROJECT_ROOT/conf/storm.yaml to $STORM_CONF_DIR/storm.yaml"
        echo "   Please check that the source file exists and that you have sufficient permissions."
        exit 1
    fi
else
    echo "⚠️  Warning: Storm conf directory not found at: $STORM_CONF_DIR"
    echo "   Please ensure Apache Storm is installed correctly and update the path in your .env file if needed."
fi

# Create a logs directory if it doesn't exist
mkdir -p "$LOG_DIR"
