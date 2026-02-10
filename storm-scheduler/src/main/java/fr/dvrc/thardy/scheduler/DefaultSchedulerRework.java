package fr.dvrc.thardy.scheduler;

import org.apache.storm.scheduler.*;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.apache.storm.metric.StormMetricsRegistry;
import org.apache.storm.utils.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/// Implements a custom scheduler based on
/// [DefaultScheduler](https://github.com/apache/storm/blob/v2.8.3/storm-server/src/main/java/org/apache/storm/scheduler/DefaultScheduler.java)
/// from Apache Storm 2.8.3.
public class DefaultSchedulerRework extends DefaultScheduler {
    private static final Logger LOG = LoggerFactory.getLogger(DefaultSchedulerRework.class);

    @Override
    public void prepare(Map<String, Object> conf, StormMetricsRegistry metricsRegistry) {
        LOG.info("DefaultSchedulerRework initialized with Storm 2.8.3 API!");
    }

    @Override
    public void schedule(Topologies topologies, Cluster cluster) {
        defaultSchedule(topologies, cluster);
    }

    @Override
    public Map<String, Map<String, Double>> config() {
        return Collections.emptyMap();
    }
}
