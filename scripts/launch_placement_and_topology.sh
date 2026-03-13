#!/bin/bash
# Script to run placement, copy scheduler CSV inputs, and submit a Storm topology.

set -euo pipefail


# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Variables
RESULTS_DIR="$PROJECT_ROOT/results"
PLACEMENT_FILE="$RESULTS_DIR/placement.csv"

resolve_path() {
    local input_path="$1"
    if [[ "$input_path" = /* ]]; then
        printf '%s\n' "$input_path"
    else
        printf '%s\n' "$PROJECT_ROOT/$input_path"
    fi
}

# Check for arguments INFRA, APP, MAPPING, optional STRATEGY
if [ $# -lt 3 ]; then
    echo "Usage: $0 <infra_properties_file> <app_properties_file> <mapping_csv_file> [strategy]"
    echo ""
    echo "Available strategies: CSP, LLM, GreedyFirstFit, GreedyFirstIterate"
    echo ""
    echo "Examples:"
    echo "  $0 python_algo/properties/Infra_5nodes_GCP.properties python_algo/properties/Appli_5comps_GCP.properties python_algo/properties/Infra_5nodes_GCP_mapping.csv"
    echo "  $0 python_algo/properties/Infra_5nodes_GCP.properties python_algo/properties/Appli_5comps_GCP.properties python_algo/properties/Infra_5nodes_GCP_mapping.csv GreedyFirstFit"
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

METRICS_FILE="$RESULTS_DIR/metrics_${STRATEGY}.csv"

# launch python algo placement
echo "===================== Running python placement algorithm (strategy: $STRATEGY) ... ===================="
# venv activation
if [ -d "$HOME/venv" ]; then
    source $HOME/venv/bin/activate
else
    echo "⚠️  Warning : Python virtual environment not found at $HOME/venv. Please ensure you have set up the virtual environment and update the path in this script if necessary."
fi
# run the placement algorithm
python "$PROJECT_ROOT/python_algo/main.py" --infra "$INFRA_FILE" --app "$APP_FILE" --strategy "$STRATEGY" --placement-csv "$PLACEMENT_FILE" --metrics-csv "$METRICS_FILE"
echo "✅ Python placement algorithm completed successfully!"
echo "===================== Copying placement results to /etc/storm/placement.csv ... ===================="
sudo cp "$PLACEMENT_FILE" /etc/storm/placement.csv
echo "✅ Placement results copied successfully!"
echo "===================== Copying mapping file to /etc/storm/mapping.csv ... ===================="
sudo cp "$MAPPING_FILE" /etc/storm/mapping.csv
echo "✅ Mapping file copied successfully!"

## Launch the topology using the properties file
echo "===================== Launching topology from properties file ... ===================="
"$SCRIPT_DIR/launch_topology_from_properties.sh" "$APP_FILE" "DeployedTopology"

echo "Done" 
