# Storm Scheduler

To run the Storm Scheduler, you need to have Apache Storm installed and set up. You can find the installation instructions on the official [Apache Storm website](https://storm.apache.org/index.html).

Version `2.8.3` was used for testing.
Ubuntu `22.04` and `24.04` was used for testing.

## Apache Storm Installation Steps

```shell
# Update package list and install Java 17 
apt-get update
apt-get install -y openjdk-17-jdk-headless wget python3 tar

# Verify Java installation
java -version

# --- 2. INSTALL ZOOKEEPER ---
apt-get install -y zookeeperd

# --- 3. DOWNLOAD & INSTALL APACHE STORM 2.8.3 ---
STORM_VER="2.8.3"
wget https://downloads.apache.org/storm/apache-storm-$STORM_VER/apache-storm-$STORM_VER.tar.gz
tar -zxf apache-storm-$STORM_VER.tar.gz
mv apache-storm-$STORM_VER /usr/local/storm

# Add binaries to PATH
export PATH=$PATH:/usr/local/storm/bin
```

## GCP

### To see UI

On master node, run the following command to forward the port for Storm UI:

```shell
gcloud compute ssh storm-nimbus \
    --project=<PROJECT_ID> \
    --zone=<ZONE> \
    -- -L 8080:localhost:8080
```
