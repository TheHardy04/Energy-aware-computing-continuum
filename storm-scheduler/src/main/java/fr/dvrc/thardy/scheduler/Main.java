package fr.dvrc.thardy.scheduler;

import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.StormSubmitter;
import org.apache.storm.topology.TopologyBuilder;

public class Main {

    public static void main(String[] args) throws Exception {

        // 1. Build the Topology (Wire your Spouts/Bolts)
        TopologyBuilder builder = new TopologyBuilder();
        builder.setSpout("sensor-spout", new SensorSpout());
        builder.setBolt("monitor-bolt", new EnergyMonitorBolt())
                .shuffleGrouping("sensor-spout");

        // 2. Configure
        Config config = new Config();
        config.setDebug(true);
        config.setNumWorkers(2); // Use 2 worker processes (relevant for Cluster/Scheduler research)

        // 3. Decide: Local vs Remote based on arguments
        if (args.length > 0 && args[0].equals("local")) {

            // --- LOCAL MODE (IntelliJ) ---
            System.out.println("Running in LOCAL mode...");
            try (LocalCluster cluster = new LocalCluster()) {
                cluster.submitTopology("test-topology", config, builder.createTopology());
                Thread.sleep(20000); // Run for 20 seconds
            }

        } else {

            // --- CLUSTER MODE (Production / Research Server) ---
            System.out.println("Submitting to CLUSTER...");
            // args[0] is usually the topology name when running via 'storm jar'
            String topologyName = (args.length > 0) ? args[0] : "production-topology";
            StormSubmitter.submitTopology(topologyName, config, builder.createTopology());
        }
    }
}