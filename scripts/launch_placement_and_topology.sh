#!/bin/bash
# End-to-end experiment launcher: deploy GCP, run placement, start telemetry, and submit Storm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/env.sh"

RESULTS_DIR="$PROJECT_ROOT/experiments/results"
RAW_RESULTS_DIR="$PROJECT_ROOT/experiments/raw"
PLACEMENT_FILE="$RESULTS_DIR/placement.csv"
TELEMETRY_SCRIPT="$PROJECT_ROOT/services/python-placement/placement/src/realtime_telemetry.py"
TELEMETRY_LOG_FILE="${TELEMETRY_LOG_FILE:-$RAW_RESULTS_DIR/telemetry.log}"
TELEMETRY_PID_FILE="${TELEMETRY_PID_FILE:-/tmp/telemetry.pid}"

resolve_path() {
    local input_path="$1"
    if [[ "$input_path" = /* ]]; then
        printf '%s\n' "$input_path"
    else
        printf '%s\n' "$PROJECT_ROOT/$input_path"
    fi
}

resolve_gcp_project_id() {
    if [ -n "${GCP_PROJECT_ID:-}" ]; then
        printf '%s\n' "$GCP_PROJECT_ID"
        return 0
    fi

    if [ -n "${GOOGLE_CLOUD_PROJECT:-}" ]; then
        printf '%s\n' "$GOOGLE_CLOUD_PROJECT"
        return 0
    fi

    if [ -n "${GCLOUD_PROJECT:-}" ]; then
        printf '%s\n' "$GCLOUD_PROJECT"
        return 0
    fi

    if command -v gcloud >/dev/null 2>&1; then
        local detected_project
        detected_project="$(gcloud config get-value project 2>/dev/null || true)"
        if [ -n "$detected_project" ] && [ "$detected_project" != "(unset)" ]; then
            printf '%s\n' "$detected_project"
            return 0
        fi
    fi

    return 1
}

stop_telemetry_if_running() {
    if [ ! -f "$TELEMETRY_PID_FILE" ]; then
        return 0
    fi

    local telemetry_pid
    telemetry_pid="$(cat "$TELEMETRY_PID_FILE" 2>/dev/null || true)"
    if [ -z "$telemetry_pid" ]; then
        rm -f "$TELEMETRY_PID_FILE"
        return 0
    fi

    if kill -0 "$telemetry_pid" 2>/dev/null; then
        echo "🧹 Stopping existing telemetry daemon (PID $telemetry_pid)..."
        kill "$telemetry_pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            if ! kill -0 "$telemetry_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$telemetry_pid" 2>/dev/null; then
            kill -9 "$telemetry_pid" 2>/dev/null || true
        fi
    fi

    rm -f "$TELEMETRY_PID_FILE"
}

start_telemetry_daemon() {
    local gcp_project_id
    if ! gcp_project_id="$(resolve_gcp_project_id)"; then
        echo "❌ Could not resolve the GCP project ID for real-time telemetry."
        echo "   Set GCP_PROJECT_ID, GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, or configure gcloud."
        exit 1
    fi

    mkdir -p "$RAW_RESULTS_DIR"
    stop_telemetry_if_running

    echo "===================== Starting real-time telemetry daemon ... ===================="
    nohup "$PYTHON_CMD" "$TELEMETRY_SCRIPT" \
        --project-id "$gcp_project_id" \
        --output-csv "$RAW_RESULTS_DIR/realtime_metrics.csv" \
        > "$TELEMETRY_LOG_FILE" 2>&1 &
    echo $! > "$TELEMETRY_PID_FILE"
    echo "✅ Telemetry daemon started (PID $(cat "$TELEMETRY_PID_FILE"))."
    echo "   Log file: $TELEMETRY_LOG_FILE"
}

if [ $# -lt 3 ]; then
    echo "Usage: $0 <infra_properties_file> <app_properties_file> <mapping_csv_file> [strategy]"
    echo ""
    echo "Available strategies: CSP, LLM, GreedyFirstFit, GreedyFirstIterate"
    echo ""
    echo "Examples:"
    echo "  $0 configs/infra/Infra_5nodes_GCP.properties configs/app/Appli_5comps_GCP.properties configs/infra/Infra_5nodes_GCP_mapping.csv"
    echo "  $0 configs/infra/Infra_5nodes_GCP.properties configs/app/Appli_5comps_GCP.properties configs/infra/Infra_5nodes_GCP_mapping.csv GreedyFirstFit"
    exit 1
fi

INFRA_FILE="$(resolve_path "$1")"
APP_FILE="$(resolve_path "$2")"
MAPPING_FILE="$(resolve_path "$3")"
STRATEGY="${4:-CSP}"

case "$STRATEGY" in
    CSP|LLM|GreedyFirstFit|GreedyFirstIterate)
        ;;
    *)
        echo "❌ Invalid strategy '$STRATEGY'."
        echo "Valid values: CSP, LLM, GreedyFirstFit, GreedyFirstIterate"
        exit 1
        ;;
esac

if [ ! -f "$INFRA_FILE" ]; then
    echo "❌ Infrastructure properties file not found: $INFRA_FILE"
    exit 1
fi

if [ ! -f "$APP_FILE" ]; then
    echo "❌ Application properties file not found: $APP_FILE"
    exit 1
fi

if [ ! -f "$MAPPING_FILE" ]; then
    echo "❌ Mapping CSV file not found: $MAPPING_FILE"
    exit 1
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "$RAW_RESULTS_DIR"

METRICS_FILE="$RESULTS_DIR/metrics_${STRATEGY}.csv"

echo "===================== Deploying GCP infrastructure from properties ... ===================="
"$PYTHON_CMD" "$PROJECT_ROOT/gcp_automations/deploy_gcp_from_properties.py" "$INFRA_FILE"
echo "✅ GCP infrastructure deployment completed successfully!"

echo "===================== Running python placement algorithm (strategy: $STRATEGY) ... ===================="
if [ -d "$HOME/venv" ]; then
    source "$HOME/venv/bin/activate"
else
    echo "⚠️  Warning: Python virtual environment not found at $HOME/venv. Please ensure you have set up the virtual environment and update the path in this script if necessary."
fi

"$PYTHON_CMD" "$PROJECT_ROOT/services/python-placement/placement/main.py" --infra "$INFRA_FILE" --app "$APP_FILE" --strategy "$STRATEGY" --placement-csv "$PLACEMENT_FILE" --metrics-csv "$METRICS_FILE"
echo "✅ Python placement algorithm completed successfully!"

echo "===================== Copying placement results to /etc/storm/placement.csv ... ===================="
sudo cp "$PLACEMENT_FILE" /etc/storm/placement.csv
echo "✅ Placement results copied successfully!"

echo "===================== Copying mapping file to /etc/storm/mapping.csv ... ===================="
sudo cp "$MAPPING_FILE" /etc/storm/mapping.csv
echo "✅ Mapping file copied successfully!"

start_telemetry_daemon

echo "===================== Launching topology from properties file ... ===================="
"$SCRIPT_DIR/launch_topology_from_properties.sh" "$APP_FILE" "DeployedTopology"

echo "Done"
