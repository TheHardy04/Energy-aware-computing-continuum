#!/bin/bash

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

TELEMETRY_PID_FILE="${TELEMETRY_PID_FILE:-/tmp/telemetry.pid}"

stop_telemetry_daemon() {
    if [ ! -f "$TELEMETRY_PID_FILE" ]; then
        echo "ℹ️  No telemetry PID file found"
        return
    fi

    local TELEMETRY_PID
    TELEMETRY_PID="$(cat "$TELEMETRY_PID_FILE" 2>/dev/null || true)"
    if [ -z "$TELEMETRY_PID" ]; then
        echo "ℹ️  Telemetry PID file is empty"
        rm -f "$TELEMETRY_PID_FILE"
        return
    fi

    if kill -0 "$TELEMETRY_PID" 2>/dev/null; then
        echo "🛑 Stopping real-time telemetry daemon: $TELEMETRY_PID"
        kill "$TELEMETRY_PID" 2>/dev/null || true
        sleep 1
        if kill -0 "$TELEMETRY_PID" 2>/dev/null; then
            echo "⚠️  Telemetry daemon still running, forcing termination"
            kill -9 "$TELEMETRY_PID" 2>/dev/null || true
        fi
        echo "✅ Telemetry daemon stopped"
    else
        echo "ℹ️  Telemetry daemon is not running"
    fi

    rm -f "$TELEMETRY_PID_FILE"
}

echo "🛑 Stopping all Storm processes..."

# Function to kill processes matching a pattern
kill_storm_process() {
    PROCESS_NAME=$1
    SEARCH_PATTERN=$2

    # Find PIDs using pgrep
    PIDS=$(pgrep -f "$SEARCH_PATTERN")

    if [ ! -z "$PIDS" ]; then
        echo "Found $PROCESS_NAME PIDs: $PIDS"
        # Kill each PID individually to ensure they're terminated
        for PID in $PIDS; do
            kill -9 "$PID" 2>/dev/null
        done
        sleep 1

        # Check if any survived
        REMAINING=$(pgrep -f "$SEARCH_PATTERN")
        if [ -z "$REMAINING" ]; then
            echo "✅ $PROCESS_NAME killed successfully"
        else
            echo "⚠️  Some $PROCESS_NAME processes still running: $REMAINING"
            # Try pkill as backup
            pkill -9 -f "$SEARCH_PATTERN"
            sleep 1
        fi
    else
        echo "ℹ️  No $PROCESS_NAME processes found"
    fi
}

# Kill supervisors - use more specific patterns
kill_storm_process "Storm Supervisor" "org.apache.storm.daemon.supervisor.Supervisor"
# Alternative pattern if the above doesn't match
kill_storm_process "Storm Supervisor (alt)" "storm supervisor"

# Kill nimbus if running
kill_storm_process "Storm Nimbus" "org.apache.storm.daemon.nimbus.Nimbus"
kill_storm_process "Storm Nimbus (alt)" "storm nimbus"

# Kill UI if running
kill_storm_process "Storm UI" "org.apache.storm.daemon.ui.UIServer"
kill_storm_process "Storm UI (alt)" "storm ui"

# Kill any remaining storm-related Java processes
echo ""
echo "Checking for any remaining Storm processes..."
STORM_PIDS=$(ps aux | grep -i storm | grep -i java | grep -v grep | awk '{print $2}')
if [ ! -z "$STORM_PIDS" ]; then
    echo "Found additional Storm Java processes: $STORM_PIDS"
    for PID in $STORM_PIDS; do
        kill -9 $PID 2>/dev/null
    done
    sleep 1
fi

# Final verification
echo ""
REMAINING=$(ps aux | grep -i storm | grep -v grep | grep -v $0)
if [ -z "$REMAINING" ]; then
    echo "✅ All Storm processes stopped successfully"
else
    echo "⚠️  Warning: Some Storm processes may still be running:"
    echo "$REMAINING"
    echo ""
    echo "Try running this command manually:"
    echo "kill -9 \$(ps aux | grep -i storm | grep -i java | awk '{print \$2}')"
fi

stop_telemetry_daemon
