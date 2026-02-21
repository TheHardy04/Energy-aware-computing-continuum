package fr.dvrc.thardy.topology;

import org.apache.storm.Config;
import org.apache.storm.StormSubmitter;
import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;
import org.apache.storm.utils.Utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * <h1>TopologyFromCSV </h1>
 * Generates Apache Storm topologies from application properties files
 * <p>
 * Reads properties files (e.g., Appli_4comps.properties, Appli_10comps_dcns.properties)
 * and creates a Storm topology with components and links matching the specification.
 * </p>
 * <h2>Properties file format:</h2>
 * <li>application.components = N (number of components)</li>
 * <li>components.requirements = {cpu,ram,lambda,mu}, {...}, ... (one per component)</li>
 * <li>links.description = {id,src,dst,bandwidth,latency}, {...}, ... (connections)</li>
 * <li>component.DZ (optional) = {componentId,nodeId}, ... (placement constraints)</li>
 *
 * <h2>Usage:</h2>
 *  <li> Local mode:  {@code java ... TopologyFromCSV path/to/properties/file.properties}</li>
 *  <li> Cluster:     {@code java ... TopologyFromCSV path/to/properties/file.properties <topologyName>}</li>
 *
 * @author Theo Hardy
 */
public class TopologyFromCSV {

    // Component specification from properties file
    static class Component {
        int id;
        int cpu;
        int ram;
        int lambda;  // data flow rate
        int mu;      // processing rate

        Component(int id, int cpu, int ram, int lambda, int mu) {
            this.id = id;
            this.cpu = cpu;
            this.ram = ram;
            this.lambda = lambda;
            this.mu = mu;
        }
    }

    // Link specification from properties file
    static class Link {
        int id;
        int src;
        int dst;
        int bandwidth;
        int maxLatency;

        Link(int id, int src, int dst, int bandwidth, int maxLatency) {
            this.id = id;
            this.src = src;
            this.dst = dst;
            this.bandwidth = bandwidth;
            this.maxLatency = maxLatency;
        }
    }

    // Placement constraint
    static class PlacementConstraint {
        int componentId;
        int nodeId;

        PlacementConstraint(int componentId, int nodeId) {
            this.componentId = componentId;
            this.nodeId = nodeId;
        }
    }

    // -------------------- SPOUT --------------------
    public static class DataSourceSpout extends BaseRichSpout {
        private org.apache.storm.spout.SpoutOutputCollector collector;
        private final Random rnd = new Random();
        private final int lambda;  // events per second
        private long nanosPerEmit;
        private long nextEmitNanos;

        public DataSourceSpout(int lambda) {
            this.lambda = lambda;
        }

        @Override
        public void open(Map<String, Object> conf, TopologyContext context,
                         org.apache.storm.spout.SpoutOutputCollector collector) {
            this.collector = collector;
            this.nanosPerEmit = 1_000_000_000L / Math.max(this.lambda, 1);
            this.nextEmitNanos = System.nanoTime();
        }

        @Override
        public void nextTuple() {
            long now = System.nanoTime();
            long remaining = nextEmitNanos - now;

            if (remaining > 0) {
                if (remaining >= 2_000_000L) {
                    Utils.sleep((int) (remaining / 1_000_000L));
                } else {
                    Thread.yield();
                }
                return;
            }

            // Emit data
            long data = rnd.nextInt(1_000_000);
            collector.emit(new Values(data));

            // Schedule next emission
            nextEmitNanos += nanosPerEmit;

            // Prevent runaway catch-up
            if (System.nanoTime() - nextEmitNanos > 1_000_000_000L) {
                nextEmitNanos = System.nanoTime();
            }
        }

        @Override
        public void declareOutputFields(OutputFieldsDeclarer declarer) {
            declarer.declare(new Fields("data"));
        }
    }

    // -------------------- BOLT --------------------
    public static class ProcessingBolt extends BaseRichBolt {
        private final int componentId;
        private final int cpuMillis;
        private final int mu;
        private OutputCollector collector;
        private static volatile long BLACKHOLE = 0;

        public ProcessingBolt(int componentId, int cpuMillis, int mu) {
            this.componentId = componentId;
            this.cpuMillis = cpuMillis;
            this.mu = mu;
        }

        @Override
        public void prepare(Map<String, Object> topoConf, TopologyContext context, OutputCollector collector) {
            this.collector = collector;
        }

        @Override
        public void execute(Tuple input) {
            long data = input.getLongByField("data");

            // Simulate CPU work
            burnCpuMillis(cpuMillis);

            // Re-emit data (pass through the pipeline)
            collector.emit(new Values(data));
        }

        private void burnCpuMillis(int millis) {
            if (millis <= 0) return;

            final long end = System.nanoTime() + (millis * 1_000_000L);
            long x = BLACKHOLE ^ 0x9E3779B97F4A7C15L;

            while (System.nanoTime() < end) {
                x = x * 1664525L + 1013904223L;
                x ^= (x << 13);
                x ^= (x >>> 7);
                x ^= (x << 17);
            }

            BLACKHOLE = x;
        }

        @Override
        public void declareOutputFields(OutputFieldsDeclarer declarer) {
            declarer.declare(new Fields("data"));
        }
    }

    // -------------------- PARSER --------------------

    /**
     * Parse braced tuples from properties file
     * Example: "{1,2,3}, {4,5,6}" -> [[1,2,3], [4,5,6]]
     */
    private static List<List<String>> parseBracedTuples(String input) {
        List<List<String>> result = new ArrayList<>();
        Pattern pattern = Pattern.compile("\\{([^}]*)\\}");
        Matcher matcher = pattern.matcher(input);

        while (matcher.find()) {
            String content = matcher.group(1);
            List<String> tuple = new ArrayList<>();
            for (String part : content.split(",")) {
                String trimmed = part.trim();
                if (!trimmed.isEmpty()) {
                    tuple.add(trimmed);
                }
            }
            if (!tuple.isEmpty()) {
                result.add(tuple);
            }
        }

        return result;
    }

    /**
     * Load application properties from file
     */
    private static Properties loadProperties(String filePath) throws IOException {
        Properties props = new Properties();
        try (InputStream input = new FileInputStream(filePath)) {
            props.load(input);
        }
        return props;
    }

    /**
     * Parse components from properties
     */
    private static List<Component> parseComponents(Properties props) {
        String requirementsStr = props.getProperty("components.requirements", "");
        List<List<String>> tuples = parseBracedTuples(requirementsStr);

        List<Component> components = new ArrayList<>();
        for (int i = 0; i < tuples.size(); i++) {
            List<String> tuple = tuples.get(i);
            if (tuple.size() >= 4) {
                int cpu = Integer.parseInt(tuple.get(0));
                int ram = Integer.parseInt(tuple.get(1));
                int lambda = Integer.parseInt(tuple.get(2));
                int mu = Integer.parseInt(tuple.get(3));
                components.add(new Component(i, cpu, ram, lambda, mu));
            }
        }

        return components;
    }

    /**
     * Parse links from properties
     */
    private static List<Link> parseLinks(Properties props) {
        String linksStr = props.getProperty("links.description", "");
        List<List<String>> tuples = parseBracedTuples(linksStr);

        List<Link> links = new ArrayList<>();
        for (List<String> tuple : tuples) {
            if (tuple.size() >= 5) {
                int id = Integer.parseInt(tuple.get(0));
                int src = Integer.parseInt(tuple.get(1));
                int dst = Integer.parseInt(tuple.get(2));
                int bandwidth = Integer.parseInt(tuple.get(3));
                int maxLatency = Integer.parseInt(tuple.get(4));
                links.add(new Link(id, src, dst, bandwidth, maxLatency));
            }
        }

        return links;
    }

    /**
     * Parse placement constraints from properties
     */
    private static List<PlacementConstraint> parseConstraints(Properties props) {
        String constraintsStr = props.getProperty("component.DZ", "");
        List<List<String>> tuples = parseBracedTuples(constraintsStr);

        List<PlacementConstraint> constraints = new ArrayList<>();
        for (List<String> tuple : tuples) {
            if (tuple.size() >= 2) {
                int componentId = Integer.parseInt(tuple.get(0));
                int nodeId = Integer.parseInt(tuple.get(1));
                constraints.add(new PlacementConstraint(componentId, nodeId));
            }
        }

        return constraints;
    }

    // -------------------- MAIN --------------------
    public static void main(String[] args) throws Exception {
        if (args == null || args.length < 1) {
            System.err.println("Usage: TopologyFromCSV <propertiesFile> [topologyName]");
            System.err.println("Example: TopologyFromCSV ../python_algo/properties/Appli_4comps.properties MyTopology");
            System.exit(1);
        }

        String propertiesFile = args[0];
        String topologyName = (args.length > 1) ? args[1] : "TopologyFromCSV";

        System.out.println("Loading application properties from: " + propertiesFile);
        Properties props = loadProperties(propertiesFile);

        // Parse application specification
        int componentsCount = Integer.parseInt(props.getProperty("application.components", "0"));
        List<Component> components = parseComponents(props);
        List<Link> links = parseLinks(props);
        List<PlacementConstraint> constraints = parseConstraints(props);

        System.out.println("Building topology with " + components.size() + " components and " + links.size() + " links");

        // Build topology
        TopologyBuilder builder = new TopologyBuilder();

        // Find source component (the one that no link points to as destination)
        Set<Integer> destinations = new HashSet<>();
        for (Link link : links) {
            destinations.add(link.dst);
        }

        int sourceComponentId = -1;
        for (Component comp : components) {
            if (!destinations.contains(comp.id)) {
                sourceComponentId = comp.id;
                break;
            }
        }

        if (sourceComponentId == -1 && !components.isEmpty()) {
            sourceComponentId = 0; // Default to first component
        }

        // Create spout for source component
        Component sourceComp = components.get(sourceComponentId);
        builder.setSpout("component_" + sourceComponentId,
                         new DataSourceSpout(sourceComp.lambda), 1);

        System.out.println("Created spout: component_" + sourceComponentId + " (lambda=" + sourceComp.lambda + ")");

        // Create bolts for other components (without connections yet)
        Map<String, org.apache.storm.topology.BoltDeclarer> boltDeclarers = new HashMap<>();
        for (Component comp : components) {
            if (comp.id != sourceComponentId) {
                // Calculate CPU work based on processing rate (mu)
                // Higher mu = more processing = more CPU work
                int cpuMillis = Math.max(1, 1000 / Math.max(comp.mu, 100));
                String boltName = "component_" + comp.id;
                org.apache.storm.topology.BoltDeclarer declarer =
                    builder.setBolt(boltName, new ProcessingBolt(comp.id, cpuMillis, comp.mu), 1);
                boltDeclarers.put(boltName, declarer);
                System.out.println("Created bolt: " + boltName +
                                 " (cpu=" + comp.cpu + ", mu=" + comp.mu + ", cpuMillis=" + cpuMillis + ")");
            }
        }

        // Wire components according to links
        for (Link link : links) {
            String srcName = "component_" + link.src;
            String dstName = "component_" + link.dst;

            // Get the bolt declarer for destination and add grouping
            org.apache.storm.topology.BoltDeclarer declarer = boltDeclarers.get(dstName);
            if (declarer != null) {
                declarer.shuffleGrouping(srcName);
                System.out.println("Connected: " + srcName + " -> " + dstName +
                                 " (bandwidth=" + link.bandwidth + ", latency=" + link.maxLatency + ")");
            }
        }

        // Apply placement constraints (using component configuration)
        Config conf = new Config();
        conf.setDebug(false);
        conf.setNumWorkers(3);
        conf.setNumAckers(0);

        // Add constraint hints to configuration
        for (PlacementConstraint constraint : constraints) {
            String key = "component." + constraint.componentId + ".node.hint";
            conf.put(key, constraint.nodeId);
            System.out.println("Constraint: component_" + constraint.componentId + " -> node " + constraint.nodeId);
        }

        // Submit topology
        System.out.println("\nSubmitting topology: " + topologyName);
        StormSubmitter.submitTopology(topologyName, conf, builder.createTopology());
        System.out.println("Topology submitted successfully!");
    }
}
