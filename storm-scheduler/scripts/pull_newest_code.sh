#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project root
cd "$SCRIPT_DIR/.."

echo "==================== Pulling Latest Code from Git... ===================="
# Reset any local changes to avoid merge conflicts
git reset --hard

# Fetch the latest changes from the remote repository
git fetch origin main

# Pull the latest code from the main branch
git pull origin main

# make the scripts executable
chmod +x "$SCRIPT_DIR"/*.sh