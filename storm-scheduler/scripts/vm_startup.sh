#!/bin/bash

## This script is intended to be run on a VM startup to set up the environment for Apache Storm.
## To run storm-scheduler on a VM, use the start_master.sh and start_worker.sh scripts

# Update package list and install Java 17
apt-get update
apt-get install -y openjdk-17-jdk-headless wget python3 tar

# Verify Java installation
java -version

# --- 2. INSTALL ZOOKEEPER ---
apt-get install -y zookeeperd

# --- 3. DOWNLOAD & INSTALL APACHE STORM 2.8.3 ---
STORM_VER="2.8.3"
wget https://downloads.apache.org/storm/apache-storm-$STORM_VER/apache-storm-$STORM_VER.tar.gz || { echo "Error: Failed to download Apache Storm $STORM_VER." >&2; exit 1; }
tar -zxf apache-storm-$STORM_VER.tar.gz || { echo "Error: Failed to extract Apache Storm archive apache-storm-$STORM_VER.tar.gz." >&2; exit 1; }
mv apache-storm-$STORM_VER /usr/local/storm

# Add binaries to PATH
echo 'export PATH=$PATH:/usr/local/storm/bin' | sudo tee -a /etc/profile.d/storm.sh > /dev/null
sudo chmod +x /etc/profile.d/storm.sh