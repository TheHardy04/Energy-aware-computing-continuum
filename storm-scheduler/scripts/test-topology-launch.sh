#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Variables
SCHEDULER_JAR="target/storm-scheduler-1.0-SNAPSHOT.jar"
TOPOLOGY_CLASS="fr.dvrc.thardy.topology.TestTopology"

# Clean and build the project using Maven
echo "==================== Building Test Topology... ===================="
mvn clean package

# Check if the build was successful
if [ $? -eq 0 ]
then
    echo "✅ Build successful! Submitting topology to Storm..."

    # Submit the topology to Storm if storm command is available
    if ! command -v storm &> /dev/null
    then
        echo "⚠️  'storm' command not found. Please ensure Apache Storm is installed and added to your PATH."
        exit 1
    fi
    storm jar "$SCHEDULER_JAR" "$TOPOLOGY_CLASS"
    if [ $? -eq 0 ]
    then
        echo "✅ Topology submitted successfully!"
    else
        echo "❌ Failed to submit topology. Storm might not be running or there could be an issue with the topology configuration."
        exit 1
    fi
else
    echo "❌ Build failed. Please check the Maven errors above."
    exit 1
fi