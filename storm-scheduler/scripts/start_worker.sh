#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables
source "$SCRIPT_DIR/env.sh"

mkdir -p "$LOG_DIR"

echo "=============== Starting Supervisor Node... ==============="
storm supervisor > "$LOG_DIR/supervisor.log" 2>&1 &
# get the PID of the last background process (the supervisor)
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