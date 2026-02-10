#!/bin/bash

# Get the absolute path of the current directory
PROJECT_ROOT=$(pwd)
CONF_ROOT="$PROJECT_ROOT/conf"
LOG_DIR="$PROJECT_ROOT/logs"

# Create a logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "=========== Launching Supervisors from $CONF_ROOT... ============"

# Loop through every folder in conf/
for dir in "$CONF_ROOT"/supervisor*/; do
    # Extract the directory name
    # One directory per supervisor, e.g. conf/supervisor1, conf/supervisor2, etc.
    SUPERVISOR_NAME=$(basename "$dir")

    echo "   Starting $SUPERVISOR_NAME on port(s) defined in storm.yaml..."

    # Set the config directory environment variable
    # Run storm supervisor in the background
    # Save logs to a separate file so they don't clutter the terminal
    STORM_CONF_DIR="$dir" storm supervisor > "$LOG_DIR/$SUPERVISOR_NAME.log" 2>&1 &
    if [ $? -eq 0 ]; then
        echo "✅ $SUPERVISOR_NAME launched successfully! Logs are being written to $LOG_DIR/$SUPERVISOR_NAME.log"
    else
        echo "❌ Failed to launch one or more supervisors. Check the logs in $LOG_DIR/ for details."
        exit 1
    fi
    # Wait a tiny bit to prevent port collisions during startup
    sleep 2
done
