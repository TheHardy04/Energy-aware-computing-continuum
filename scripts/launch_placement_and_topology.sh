#!/bin/bash
# End-to-end experiment launcher for a pre-deployed Storm Nimbus master.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STORM_SCHEDULER_DIR="$PROJECT_ROOT/services/java-storm-scheduler"
TELEMETRY_SCRIPT="$PROJECT_ROOT/services/python-placement/placement/src/realtime_telemetry.py"
SCHEDULER_JAR="$STORM_SCHEDULER_DIR/target/storm-scheduler-1.0-SNAPSHOT.jar"
TOPOLOGY_CLASS="fr.dvrc.thardy.topology.TopologyFromProperties"
TOPOLOGY_NAME="DeployedTopology"

cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/env.sh"

RESULTS_DIR="$PROJECT_ROOT/experiments/results"
RAW_RESULTS_DIR="$PROJECT_ROOT/experiments/raw"
PLACEMENT_FILE="$RESULTS_DIR/placement.csv"
TELEMETRY_LOG_FILE="$RAW_RESULTS_DIR/telemetry.log"
TELEMETRY_PID_FILE="/tmp/telemetry.pid"

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
        echo "Stopping existing telemetry daemon with PID $telemetry_pid..."
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

needs_scheduler_build() {
    if [ ! -f "$SCHEDULER_JAR" ]; then
        return 0
    fi

    if find "$STORM_SCHEDULER_DIR/src/main/java" -type f -newer "$SCHEDULER_JAR" -print -quit | grep -q .; then
        return 0
    fi

    if find "$STORM_SCHEDULER_DIR/pom.xml" -newer "$SCHEDULER_JAR" -print -quit | grep -q .; then
        return 0
    fi

    return 1
}

build_scheduler_if_needed() {
    if needs_scheduler_build; then
        echo "Building the Storm scheduler module with Maven..."
        (cd "$STORM_SCHEDULER_DIR" && mvn clean package)
        if [ ! -f "$SCHEDULER_JAR" ]; then
            echo "Failed to build the Storm scheduler JAR: $SCHEDULER_JAR"
            exit 1
        fi
        echo "Storm scheduler build completed successfully."
    else
        echo "Storm scheduler JAR is up to date."
    fi
}

start_telemetry_daemon() {
    local gcp_project_id
    if ! gcp_project_id="$(resolve_gcp_project_id)"; then
        echo "Could not resolve the GCP project ID for real-time telemetry."
        echo "Set GCP_PROJECT_ID, GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, or configure gcloud."
        exit 1
    fi

    mkdir -p "$RAW_RESULTS_DIR"
    stop_telemetry_if_running

    echo "Starting real-time telemetry daemon..."
    nohup "$PYTHON_CMD" "$TELEMETRY_SCRIPT" \
        --project-id "$gcp_project_id" \
        --output-csv "$RAW_RESULTS_DIR/realtime_metrics.csv" \
        > "$TELEMETRY_LOG_FILE" 2>&1 &
    echo $! > "$TELEMETRY_PID_FILE"
    echo "Telemetry daemon started with PID $(cat "$TELEMETRY_PID_FILE")."
    echo "Log file: $TELEMETRY_LOG_FILE"
}

submit_topology() {
    if ! command -v storm >/dev/null 2>&1; then
        echo "The 'storm' command was not found. Ensure Apache Storm is installed and available on PATH."
        exit 1
    fi

    echo "Submitting the topology to local Nimbus..."
    storm jar "$SCHEDULER_JAR" "$TOPOLOGY_CLASS" "$APP_FILE" "$TOPOLOGY_NAME"
    echo "Topology '$TOPOLOGY_NAME' submitted successfully."
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
export TELEMETRY_LOG_FILE
export TELEMETRY_PID_FILE

METRICS_FILE="$RESULTS_DIR/metrics_${STRATEGY}.csv"

echo "Running the Python placement algorithm with strategy $STRATEGY..."

"$PYTHON_CMD" "$PROJECT_ROOT/services/python-placement/placement/main.py" --infra "$INFRA_FILE" --app "$APP_FILE" --strategy "$STRATEGY" --placement-csv "$PLACEMENT_FILE" --metrics-csv "$METRICS_FILE"
echo "Python placement completed successfully."

echo "Copying placement results to /etc/storm/placement.csv..."
sudo cp "$PLACEMENT_FILE" /etc/storm/placement.csv
echo "Placement results copied successfully."

echo "Copying mapping file to /etc/storm/mapping.csv..."
sudo cp "$MAPPING_FILE" /etc/storm/mapping.csv
echo "Mapping file copied successfully."

build_scheduler_if_needed
start_telemetry_daemon
submit_topology

echo "Experiment launch completed."
