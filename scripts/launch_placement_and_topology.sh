#!/bin/bash
# Script to launch the placement and topology in the cluster after the VMs are setup and storm is running


# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Variables

# Check for arguments INFRA, APP, MAPPING
if [ $# -lt 3 ]; then
    echo "Usage: $0 <infra_properties_file> <app_properties_file> <mapping_csv_file>"
    echo ""
    echo "Example:"
    echo "  $0 python_algo/properties/Infra_5nodes_GCP.properties python_algo/properties/Appli_4comps.properties python_algo/properties/Infra_5nodes_GCP_mapping.csv"
    exit 1
fi

# launch python algo placement
echo "===================== Running python placement algorithm ... ===================="
# venv activation
source "$PROJECT_ROOT/venv/bin/activate"
# run the placement algorithm
python "$PROJECT_ROOT/python_algo/main.py" --infra "$1" --app "$2" --to-csv "$PROJECT_ROOT/result/placement.csv"
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
sudo cp "$3" /etc/storm/mapping.csv
if [ $? -ne 0 ]; then
    echo "❌ Failed to copy mapping file to /etc/storm/mapping.csv. Please check permissions and ensure the path is correct."
    exit 1
fi
echo "✅ Mapping file copied successfully!"

## Launch the topology using the properties file
echo "===================== Launching topology from properties file ... ===================="
"$SCRIPT_DIR/launch_topology_from_properties.sh" "$2" "DeployedTopology"
if [ $? -ne 0 ]; then
    echo "❌ Failed to launch topology from properties file. Please check the errors above."
    exit 1
fi

echo "Done" 
