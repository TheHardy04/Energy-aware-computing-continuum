#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

## Variables
SCHEDULER_JAR="target/storm-scheduler-1.0-SNAPSHOT-all.jar"

# Load environment variables
source "$SCRIPT_DIR/env.sh"

# Clean and build the project using Maven
echo "==================== Building Schedulers... ===================="
mvn clean package

# Check if the build was successful
if [ $? -eq 0 ]
then
    echo "✅ Build successful! Deploying schedulers to Storm..."
    # Copy generated JAR to Storm's lib directory
    if [ -d "$STORM_LIB_DIR" ]; then
        cp "$SCHEDULER_JAR" "$STORM_LIB_DIR"
        if [ $? -eq 0 ]; then
            echo "✅ Schedulers deployed successfully!"
            echo "You can now use the schedulers in your Storm topologies by specifying it in the configuration."
            echo "Please restart your Storm cluster to ensure the new scheduler is loaded."
        else
            echo "❌ Failed to copy JAR to Storm's lib directory. Please check permissions and ensure the path specified is correct."
            exit 1
        fi
    else
        echo "⚠️  Storm lib directory not found. Please ensure Apache Storm is installed and update the path in this script."
        exit 1
    fi
else
    echo "❌ Build failed. Please check the Maven errors above."
    exit 1
fi