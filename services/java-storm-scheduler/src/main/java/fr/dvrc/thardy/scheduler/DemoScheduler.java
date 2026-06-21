package fr.dvrc.thardy.scheduler;

import org.apache.storm.scheduler.*;
import org.apache.storm.metric.StormMetricsRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

/// A simple custom scheduler that implements the
/// [IScheduler interface](https://github.com/apache/storm/blob/v2.8.3/storm-server/src/main/java/org/apache/storm/scheduler/IScheduler.java)
/// on Apache Storm 2.8.3.
public class DemoScheduler implements IScheduler {
    private static final Logger LOG = LoggerFactory.getLogger(DemoScheduler.class);
    private Map<String, Object> conf;

    @Override
    public void prepare(Map<String, Object> conf, StormMetricsRegistry metricsRegistry) {
        // Save the configuration for later use
        this.conf = conf;
        LOG.info("DemoScheduler initialized with Storm 2.8.3 API!");
    }


    @Override
    /**
     * Set assignments for the topologies which needs scheduling. The new assignments is available
     * through `cluster.getAssignments()`
     *
     *@param topologies all the topologies in the cluster, some of them need schedule. Topologies object here
     *       only contain static information about topologies. Information like assignments, slots are all in
     *       the `cluster` object.
     *@param cluster the cluster these topologies are running in. `cluster` contains everything user
     *       need to develop a new scheduling logic. e.g. supervisors information, available slots, current
     *       assignments for all the topologies etc. User can set the new assignment for topologies using
     *       cluster.setAssignmentById()`
     */
    public void schedule(Topologies topologies, Cluster cluster) {
        LOG.info("DemoScheduler: Scheduling cycle started...");

        // 1. Get topologies that need scheduling
        List<TopologyDetails> needsScheduling = cluster.needsSchedulingTopologies();

        for (TopologyDetails topology : needsScheduling) {
            LOG.info("Scheduling topology: {}", topology.getName());

            // 2. Get all executors (tasks) that need assignment
            // Map<ComponentId, List<ExecutorDetails>>
            Map<String, List<ExecutorDetails>> componentToExecutors =
                    cluster.getNeedsSchedulingComponentToExecutors(topology);

            List<ExecutorDetails> allExecutors = new ArrayList<>();
            for (List<ExecutorDetails> execs : componentToExecutors.values()) {
                allExecutors.addAll(execs);
            }

            // 3. Get available worker slots (ports on supervisors)
            List<WorkerSlot> availableSlots = cluster.getAvailableSlots();

            if (availableSlots.isEmpty()) {
                LOG.warn("No slots available to schedule {}", topology.getName());
                continue;
            }

            // 4. Simple Round-Robin Assignment
            int numSlots = availableSlots.size();
            for (int i = 0; i < allExecutors.size(); i++) {
                ExecutorDetails executor = allExecutors.get(i);
                WorkerSlot targetSlot = availableSlots.get(i % numSlots);

                // Assign the executor to the slot
                if(!cluster.isSlotOccupied(targetSlot)) {
                    cluster.assign(targetSlot, topology.getId(), Collections.singletonList(executor));
                }
                else {
                    LOG.warn("Slot {} is already occupied, skipping assignment for executor {}", targetSlot, executor);
                }

                LOG.info("Assigned executor {} to slot {}", executor, targetSlot);
            }
        }
    }

    @Override
    /**
     * This function returns the scheduler's configuration.
     *
     * @return The scheduler's configuration.
     */
    public Map<String, Object> config() {
        // Return the configuration map used by this scheduler
        return this.conf;    }

    @Override
    /**
     * called once when the system is shutting down, should be idempotent.
     */
    public void cleanup() {
    }
}
