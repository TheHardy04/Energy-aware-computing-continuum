#!/bin/bash
# Script to launch the placement and topology in the cluster after the VMs are setup and storm is running


# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Variables

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

INFRA_FILE="$1"
APP_FILE="$2"
MAPPING_FILE="$3"
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

# launch python algo placement
echo "===================== Running python placement algorithm (strategy: $STRATEGY) ... ===================="
# venv activation
if [ -d "$HOME/venv" ]; then
    source $HOME/venv/bin/activate
else
    echo "⚠️  Warning : Python virtual environment not found at $HOME/venv. Please ensure you have set up the virtual environment and update the path in this script if necessary."
fi
# run the placement algorithm
python "$PROJECT_ROOT/python_algo/main.py" --infra "$INFRA_FILE" --app "$APP_FILE" --strategy "$STRATEGY" --placement-csv "$PROJECT_ROOT/result/placement.csv" --metrics-csv "$PROJECT_ROOT/result/metrics_${STRATEGY}.csv"
if [ $? -ne 0 ]; then
    echo "❌ Python placement algorithm failed. Please check the errors above."
    exit 1
fi
echo "✅ Python placement algorithm completed successfully!"
echo "===================== Copying placement results to /etc/storm/placement.csv ... ===================="
sudo cp "$PROJECT_ROOT/result/placement.csv" /etc/storm/placement.csv
if [ $? -ne 0 ]; then
    echo "❌ Failed to copy placement results to /etc/storm/placement.csv. Please check permissions and ensure the path is correct."
    exit 1
fi
echo "✅ Placement results copied successfully!"
echo "===================== Copying mapping file to /etc/storm/mapping.csv ... ===================="
sudo cp "$MAPPING_FILE" /etc/storm/mapping.csv
if [ $? -ne 0 ]; then
    echo "❌ Failed to copy mapping file to /etc/storm/mapping.csv. Please check permissions and ensure the path is correct."
    exit 1
fi
echo "✅ Mapping file copied successfully!"

## Launch the topology using the properties file
echo "===================== Launching topology from properties file ... ===================="
"$SCRIPT_DIR/launch_topology_from_properties.sh" "$APP_FILE" "DeployedTopology"
if [ $? -ne 0 ]; then
    echo "❌ Failed to launch topology from properties file. Please check the errors above."
    exit 1
fi

echo "Done" 
