#!/bin/bash

## This script is intended to be run on a VM startup to set up the environment for Apache Storm.
## To run storm-scheduler on a VM, use the start_master.sh and start_worker.sh scripts

# Update package list and install Java 17 and other necessary tools
apt-get install -y openjdk-17-jdk-headless wget python3 tar git maven python3-pip vim

# Verify Java installation
java -version

# --- 2. INSTALL ZOOKEEPER ---
apt-get install -y zookeeperd

# --- 3. DOWNLOAD & INSTALL APACHE STORM 2.8.3 ---
STORM_VER="2.8.3"
wget https://downloads.apache.org/storm/apache-storm-$STORM_VER/apache-storm-$STORM_VER.tar.gz || { echo "Error: Failed to download Apache Storm $STORM_VER." >&2; exit 1; }
tar -zxf apache-storm-$STORM_VER.tar.gz || { echo "Error: Failed to extract Apache Storm archive apache-storm-$STORM_VER.tar.gz." >&2; exit 1; }
mv apache-storm-$STORM_VER /usr/local/storm

# --- 4. SET PERMISSIONS AND PATHS ---
# Give the right to everyone to read write and execute the storm directory (for simplicity in this VM setup)
chmod -R 777 /usr/local/storm
mkdir -p /usr/local/storm/logs

# Add binaries to PATH
ln -s /usr/local/storm/bin/storm /usr/bin/storm