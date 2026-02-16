#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables
source "$SCRIPT_DIR/env.sh"

mkdir -p "$LOG_DIR"

echo "=============== Starting Supervisor Node... ==============="
storm supervisor > "$LOG_DIR/supervisor.log" 2>&1 &
if [ $? -eq 0 ]; then
    echo "✅ Supervisor launched successfully! Logs are being written to $LOG_DIR/supervisor.log"
else
    echo "❌ Failed to launch the supervisor. Check the logs in $LOG_DIR/ for details."
    exit 1
fi

