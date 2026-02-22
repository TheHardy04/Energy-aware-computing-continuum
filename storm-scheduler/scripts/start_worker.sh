#!/bin/bash

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