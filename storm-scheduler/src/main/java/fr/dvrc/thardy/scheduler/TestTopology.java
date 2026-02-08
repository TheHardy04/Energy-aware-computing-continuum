package fr.dvrc.thardy.scheduler;

import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
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

import java.util.Map;
import java.util.Random;

/**
 * Apache Storm 2.8.3 - No-ack synthetic topology:
 * - Spout emits 1 random number ("n") at a configurable fixed rate (events/sec)
 * - 6 synthetic bolts chained, each burns a different amount of CPU per tuple
 * - Reliability disabled: TOPOLOGY_ACKERS = 0, spout emits without messageId, bolts do not ack
 *
 * Run:
 *   Local:
 *     java ... RandomNumber6BoltsCpuRateTopology
 *     java ... RandomNumber6BoltsCpuRateTopology local 100
 *
 *   Cluster:
 *     java ... RandomNumber6BoltsCpuRateTopology <topologyName> 100
 */
public class TestTopology {

    // ---------- Spout rate configuration ----------
    private static final String SPOUT_RATE_CONF = "spout.rate.eps";
    private static final int DEFAULT_RATE_EPS = 10; // default: 100 events/sec

    // ---------- CPU burner (busy spin) ----------
    // Volatile sink to prevent the JIT from optimizing away the loop.
    private static volatile long BLACKHOLE = 0;

    private static void burnCpuMillis(int millis) {
        if (millis <= 0) return;

        final long end = System.nanoTime() + (millis * 1_000_000L);
        long x = BLACKHOLE ^ 0x9E3779B97F4A7C15L;

        // Tight loop with arithmetic to keep CPU busy.
        while (System.nanoTime() < end) {
            x = x * 1664525L + 1013904223L;
            x ^= (x << 13);
            x ^= (x >>> 7);
            x ^= (x << 17);
        }

        BLACKHOLE = x;
    }

    // -------------------- SPOUT --------------------
    public static class RandomNumberRateSpout extends BaseRichSpout {
        private org.apache.storm.spout.SpoutOutputCollector collector;
        private final Random rnd = new Random();

        private int rateEps;
        private long nanosPerEmit;
        private long nextEmitNanos;

        @Override
        public void open(Map<String, Object> conf, TopologyContext context,
                         org.apache.storm.spout.SpoutOutputCollector collector) {
            this.collector = collector;

            Object v = conf.get(SPOUT_RATE_CONF);
            int r = DEFAULT_RATE_EPS;
            if (v instanceof Number) {
                r = ((Number) v).intValue();
            } else if (v instanceof String) {
                try { r = Integer.parseInt((String) v); } catch (Exception ignored) {}
            }
            if (r <= 0) r = DEFAULT_RATE_EPS;

            this.rateEps = r;
            this.nanosPerEmit = 1_000_000_000L / this.rateEps;
            this.nextEmitNanos = System.nanoTime(); // start immediately
        }

        @Override
        public void nextTuple() {
            long now = System.nanoTime();
            long remaining = nextEmitNanos - now;

            if (remaining > 0) {
                // We only use sleep/yield for pacing (this does NOT burn CPU).
                // Sleep if we're far enough; otherwise just yield.
                if (remaining >= 2_000_000L) { // >= 2ms
                    Utils.sleep((int) (remaining / 1_000_000L));
                } else {
                    Thread.yield();
                }
                return;
            }

            // Emit ONE tuple per nextTuple() call to avoid long blocking loops.
            long n = rnd.nextInt(1_000_000);
            collector.emit(new Values(n)); // no messageId => no ack tracking

            // Schedule next emission time (fixed rate).
            nextEmitNanos += nanosPerEmit;

            // If we are very late (e.g., GC pause), avoid runaway catch-up bursts:
            // clamp nextEmitNanos to now to resume smoothly.
            long lateBy = System.nanoTime() - nextEmitNanos;
            if (lateBy > 1_000_000_000L) { // > 1 second late
                nextEmitNanos = System.nanoTime();
            }
        }

        @Override
        public void declareOutputFields(OutputFieldsDeclarer declarer) {
            declarer.declare(new Fields("n"));
        }
    }

    // -------------------- BOLTS --------------------
    public static class SyntheticCpuBolt extends BaseRichBolt {
        private final int cpuMillisPerTuple;
        private OutputCollector collector;

        public SyntheticCpuBolt(int cpuMillisPerTuple) {
            this.cpuMillisPerTuple = cpuMillisPerTuple;
        }

        @Override
        public void prepare(Map<String, Object> topoConf, TopologyContext context, OutputCollector collector) {
            this.collector = collector;
        }

        @Override
        public void execute(Tuple input) {
            long n = input.getLongByField("n");

            // Burn CPU to simulate work (different per bolt)
            burnCpuMillis(cpuMillisPerTuple);

            // Re-emit the same schema (single field "n")
            collector.emit(new Values(n));

            // No ack/fail: reliability disabled (ackers=0)
        }

        @Override
        public void declareOutputFields(OutputFieldsDeclarer declarer) {
            declarer.declare(new Fields("n"));
        }
    }

    // -------------------- MAIN --------------------
    public static void main(String[] args) throws Exception {
        // Args:
        //   - Cluster: <topologyName> [rateEps]
        //   - Local:   (no args) OR "local" [rateEps]
        boolean clusterMode = (args != null && args.length > 0 && args[0] != null && !args[0].trim().isEmpty()
                && !"local".equalsIgnoreCase(args[0].trim()));

        String topologyName = clusterMode ? args[0].trim() : TestTopology.class.getSimpleName();

        int rateEps = DEFAULT_RATE_EPS;
        if (args != null && args.length > 0) {
            // rate can be args[1] in cluster mode, or args[1] in "local" mode
            int idx = clusterMode ? 1 : 1;
            if (args.length > idx) {
                try { rateEps = Integer.parseInt(args[idx].trim()); } catch (Exception ignored) {}
            }
        }
        if (rateEps <= 0) rateEps = DEFAULT_RATE_EPS;

        TopologyBuilder builder = new TopologyBuilder();

        builder.setSpout("spout", new RandomNumberRateSpout(), 1);

        // Per-bolt CPU burn (ms) example: 10ms, 5ms, 40ms, etc.
        int[] cpuMs = new int[] { 10, 5, 40, 15, 20, 2 };

        String prev = "spout";
        for (int i = 1; i <= 6; i++) {
            String boltName = "bolt" + i;
            int burn = cpuMs[i - 1];

            builder.setBolt(boltName, new SyntheticCpuBolt(burn), 1)
                    .shuffleGrouping(prev);

            prev = boltName;
        }

        Config conf = new Config();
        conf.setDebug(false);
        conf.setNumWorkers(3);
        conf.setNumAckers(0);

        // Pass spout rate as a parameter (config)
        conf.put(SPOUT_RATE_CONF, rateEps);

        if (clusterMode) {
            StormSubmitter.submitTopology(topologyName, conf, builder.createTopology());
        } else {
            LocalCluster cluster = new LocalCluster();
            cluster.submitTopology(topologyName, conf, builder.createTopology());

            // Run for 60 seconds, then shut down
            Utils.sleep(60_000);

            cluster.killTopology(topologyName);
            cluster.shutdown();
        }
    }
}