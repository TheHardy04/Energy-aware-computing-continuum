#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Variables
SCHEDULER_JAR="target/storm-scheduler-1.0-SNAPSHOT.jar"
TOPOLOGY_CLASS="fr.dvrc.thardy.topology.TopologyFromCSV"

# Check for arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <properties_file> [topology_name]"
    echo ""
    echo "Example:"
    echo "  $0 ../python_algo/properties/Appli_4comps.properties MyTopology"
    echo "  $0 ../python_algo/properties/Appli_10comps_dcns.properties DCNS"
    exit 1
fi

PROPERTIES_FILE="$1"
TOPOLOGY_NAME="${2:-TopologyFromCSV}"

# Check if properties file exists
if [ ! -f "$PROPERTIES_FILE" ]; then
    echo "❌ Error: Properties file not found: $PROPERTIES_FILE"
    exit 1
fi

# Clean and build the project using Maven
echo "==================== Building TopologyFromCSV... ===================="
mvn clean package

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please check the Maven errors above."
    exit 1
fi

echo "✅ Build successful! Submitting topology to Storm..."

# Check if the JAR exists
if [ ! -f "$SCHEDULER_JAR" ]; then
    echo "❌ Error: JAR file not found: $SCHEDULER_JAR"
    exit 1
fi

# Submit the topology to Storm if storm command is available
if ! command -v storm &> /dev/null; then
    echo "⚠️  'storm' command not found. Please ensure Apache Storm is installed and added to your PATH."
    exit 1
fi

echo "Submitting topology with properties from: $PROPERTIES_FILE"
storm jar "$SCHEDULER_JAR" "$TOPOLOGY_CLASS" "$PROPERTIES_FILE" "$TOPOLOGY_NAME"

if [ $? -eq 0 ]; then
    echo "✅ Topology '$TOPOLOGY_NAME' submitted successfully!"
else
    echo "❌ Failed to submit topology. Storm might not be running or there could be an issue with the topology configuration."
    exit 1
fi

