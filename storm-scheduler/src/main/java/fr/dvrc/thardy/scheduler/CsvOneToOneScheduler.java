package fr.dvrc.thardy.scheduler;

import com.opencsv.CSVReader;
import org.apache.storm.metric.StormMetricsRegistry;
import org.apache.storm.scheduler.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class CsvOneToOneScheduler implements IScheduler {

    private static final Logger LOG = LoggerFactory.getLogger(CsvOneToOneScheduler.class);

    private String csvFile;
    private boolean hasHeader;

    @Override
    public void prepare(Map<String, Object> conf, StormMetricsRegistry metricsRegistry) {
        Object configuredFile = conf.get("csv.scheduler.file");
        if (configuredFile != null) {
            this.csvFile = String.valueOf(configuredFile);
        } else {
            String stormHome = System.getenv("STORM_HOME");
            if (stormHome != null && !stormHome.isBlank()) {
                this.csvFile = stormHome + "/conf/component-placement.csv";
            } else {
                this.csvFile = "component-placement.csv";
            }
        }
        this.hasHeader = Boolean.parseBoolean(String.valueOf(conf.getOrDefault(
                "csv.scheduler.hasHeader",
                "false"
        )));

        LOG.info("CsvOneToOneScheduler prepared. csvFile={}, hasHeader={}", csvFile, hasHeader);
    }

    @Override
    public void schedule(Topologies topologies, Cluster cluster) {
        // component -> host (or supervisorId)
        Map<String, String> placement = readPlacementCsv(csvFile, hasHeader);

        for (TopologyDetails topology : topologies.getTopologies()) {
            if (!cluster.needsScheduling(topology)) {
                continue;
            }

            LOG.info("Scheduling topology: {} ({})", topology.getName(), topology.getId());

            // component -> executors that still need scheduling
            Map<String, List<ExecutorDetails>> needs =
                    new HashMap<>(cluster.getNeedsSchedulingComponentToExecutors(topology));

            if (needs.isEmpty()) {
                LOG.info("Nothing to schedule for topology {}", topology.getName());
                continue;
            }

            // 1) Group executors by target host according to CSV
            // host -> list of executors that should run there
            Map<String, List<ExecutorDetails>> hostToExecs = new LinkedHashMap<>();

            for (Map.Entry<String, String> rule : placement.entrySet()) {
                String component = rule.getKey();
                String hostOrId = rule.getValue();

                List<ExecutorDetails> execs = needs.get(component);
                if (execs == null || execs.isEmpty()) {
                    continue; // component not in this topology or already scheduled
                }

                hostToExecs.computeIfAbsent(hostOrId, k -> new ArrayList<>()).addAll(execs);
                needs.remove(component);
            }

            // 2) Assign ONE worker slot per host, packing all its executors into that slot
            Set<WorkerSlot> usedSlots = new LinkedHashSet<>();

            for (Map.Entry<String, List<ExecutorDetails>> entry : hostToExecs.entrySet()) {
                String hostOrId = entry.getKey();
                List<ExecutorDetails> execsForHost = entry.getValue();

                SupervisorDetails sup = resolveSupervisor(cluster, hostOrId);
                if (sup == null) {
                    LOG.warn("No supervisor found for '{}'. Those executors remain unscheduled.", hostOrId);
                    continue;
                }

                List<WorkerSlot> slots = cluster.getAvailableSlots(sup);
                if (slots == null || slots.isEmpty()) {
                    LOG.warn("No available worker slots on host={} id={}", sup.getHost(), sup.getId());
                    continue;
                }

                WorkerSlot chosen = slots.get(0); // ONE slot per VM/host
                cluster.assign(chosen, topology.getId(), execsForHost);
                usedSlots.add(chosen);

                LOG.info("Packed {} executors into ONE worker on host={} port={}",
                        execsForHost.size(), sup.getHost(), chosen.getPort());
            }

            // 3) Fallback: pack remaining components into already-created workers (do NOT create more workers)
            if (!needs.isEmpty()) {
                List<WorkerSlot> slotList = new ArrayList<>(usedSlots);

                // If nothing was assigned yet, fallback to any free slots (not ideal, but prevents stalling)
                if (slotList.isEmpty()) {
                    slotList = new ArrayList<>(cluster.getAvailableSlots());
                }

                int idx = 0;
                for (Map.Entry<String, List<ExecutorDetails>> rem : needs.entrySet()) {
                    if (slotList.isEmpty()) {
                        LOG.warn("No slots available for remaining components: {}", needs.keySet());
                        break;
                    }

                    WorkerSlot target = slotList.get(idx % slotList.size());
                    cluster.assign(target, topology.getId(), rem.getValue());

                    LOG.info("Fallback packed component '{}' into existing worker nodeId={} port={}",
                            rem.getKey(), target.getNodeId(), target.getPort());

                    idx++;
                }
            }
        }
    }

    @Override
    public Map config() {
        return Collections.emptyMap();
    }

    @Override
    public void cleanup() {
        // no-op
    }

    private Map<String, String> readPlacementCsv(String file, boolean hasHeader) {
        Map<String, String> out = new LinkedHashMap<>();

        try {
            Path p = Path.of(file);
            if (!Files.exists(p)) {
                LOG.warn("Placement CSV does not exist: {}", file);
                return out;
            }

            try (CSVReader reader = new CSVReader(new FileReader(file))) {
                String[] row;
                boolean first = true;

                while ((row = reader.readNext()) != null) {
                    if (row.length < 2) continue;

                    if (first && hasHeader) {
                        first = false;
                        continue;
                    }
                    first = false;

                    String component = row[0] == null ? "" : row[0].trim();
                    String hostOrId = row[1] == null ? "" : row[1].trim();

                    if (component.isEmpty() || hostOrId.isEmpty()) continue;
                    if (component.startsWith("#")) continue; // allow comments

                    out.put(component, hostOrId);
                }
            }

            LOG.info("Loaded {} placement rules from CSV {}", out.size(), file);
        } catch (Exception ex) {
            LOG.error("Error reading placement CSV: " + file, ex);
        }

        return out;
    }

    private SupervisorDetails resolveSupervisor(Cluster cluster, String hostOrId) {
        // 1) Exact match by Supervisor ID
        SupervisorDetails byId = cluster.getSupervisorById(hostOrId);
        if (byId != null) return byId;

        // 2) Exact match by host name
        for (SupervisorDetails s : cluster.getSupervisors().values()) {
            if (hostOrId.equals(s.getHost())) return s;
        }

        // 3) Soft match (vm1 vs vm1.domain, etc.)
        for (SupervisorDetails s : cluster.getSupervisors().values()) {
            String h = s.getHost();
            if (h != null && (h.startsWith(hostOrId) || h.contains(hostOrId))) {
                return s;
            }
        }

        return null;
    }
}