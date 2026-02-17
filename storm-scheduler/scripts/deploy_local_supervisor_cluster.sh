#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

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
    SUPERVISOR_PID=$!
    # Wait a moment to allow the supervisor to start and write logs
    sleep 2
    # Check if the supervisor process is still running
    if ps -p $SUPERVISOR_PID > /dev/null; then
        echo "Supervisor started successfully (PID: $SUPERVISOR_PID)"
    else
        echo "Failed to start supervisor. Check $LOG_DIR/supervisor.log for details"
        exit 1
    fi

done
