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
                "true"
        )));

        LOG.info("CsvOneToOneScheduler prepared. csvFile={}, hasHeader={}", csvFile, hasHeader);
    }

    @Override
    public void schedule(Topologies topologies, Cluster cluster) {
        // component -> host (or supervisorId)
        Map<String, String> placement = readPlacementCsv(csvFile, hasHeader);

        // Load VM mapping if available (host ID -> VM name)
        Map<String, String> hostIdToVmName = loadHostMapping(csvFile);

        // Log available supervisors for debugging
        logAvailableSupervisors(cluster);

        for (TopologyDetails topology : topologies.getTopologies()) {
            if (!cluster.needsScheduling(topology)) {
                continue;
            }

            scheduleTopology(cluster, topology, placement, hostIdToVmName);
        }
    }

    private void scheduleTopology(Cluster cluster, TopologyDetails topology,
                                  Map<String, String> placement,
                                  Map<String, String> hostIdToVmName) {
        LOG.info("Scheduling topology: {} ({})", topology.getName(), topology.getId());

        // 0) Pre-processing: Unassign executors that are currently placed on the WRONG host.
        // This ensures that "Available Slots" are actually freed up for the correct placement.
        unassignImproperlyPlacedExecutors(cluster, topology, placement, hostIdToVmName);

        // component -> executors that still need scheduling
        // (This will now include the ones we just unassigned)
        Map<String, List<ExecutorDetails>> needs =
                new HashMap<>(cluster.getNeedsSchedulingComponentToExecutors(topology));

        if (needs.isEmpty()) {
            LOG.info("Nothing to schedule for topology {}", topology.getName());
            return;
        }

        // 1) Group executors by target host according to CSV
        // host -> list of executors that should run there
        Map<String, List<ExecutorDetails>> hostToExecs = new LinkedHashMap<>();
        // host -> list of component names grouped for that host
        Map<String, List<String>> hostToComponents = new LinkedHashMap<>();

        groupExecutorsByHost(needs, placement, hostToExecs, hostToComponents);

        // 2) Assign ONE worker slot per host, packing all its executors into that slot
        Set<WorkerSlot> usedSlots = assignExecutorsFromCsv(cluster, topology, hostToExecs, hostToComponents, needs, hostIdToVmName);

        // 3) Fallback: pack remaining components into available slots
        if (!needs.isEmpty()) {
            assignRemainingExecutors(cluster, topology, needs, usedSlots);
        }
    }

    private void unassignImproperlyPlacedExecutors(Cluster cluster, TopologyDetails topology,
                                                   Map<String, String> placement,
                                                   Map<String, String> hostIdToVmName) {
        // SchedulerAssignment contains the current assignment for this topology
        SchedulerAssignment assignment = cluster.getAssignmentById(topology.getId());
        if (assignment == null) {
            return; // Not scheduled yet
        }

        Map<ExecutorDetails, WorkerSlot> executorToSlot = assignment.getExecutorToAssignment();
        if (executorToSlot == null) return;
        
        Set<ExecutorDetails> executorsToUnassign = new HashSet<>();

        for (Map.Entry<ExecutorDetails, WorkerSlot> entry : executorToSlot.entrySet()) {
            ExecutorDetails executor = entry.getKey();
            WorkerSlot slot = entry.getValue();

            String componentId = topology.getExecutorToComponent().get(executor);
            if (componentId == null) continue;
            
            // Determine desired placement
            String targetHostOrId = getTargetHost(componentId, placement);
            if (targetHostOrId == null) {
                // No rule for this component, ignore it (keep where it is)
                continue;
            }

            SupervisorDetails desiredSupervisor = resolveSupervisor(cluster, targetHostOrId, hostIdToVmName);
            if (desiredSupervisor == null) {
                // If the target doesn't exist, we can't move it there.
                // Keep it where it is or let fallback handle?
                // For now, keep it where it is to avoid "unscheduled" state if no fallback works.
                continue;
            }

            // Check if current slot is on the desired supervisor
            if (!slot.getNodeId().equals(desiredSupervisor.getId())) {
                LOG.info("Executor {} (component {}) is on wrong supervisor {}. Target is {}. Unassigning...",
                        executor, componentId, slot.getNodeId(), desiredSupervisor.getId());
                executorsToUnassign.add(executor);
            }
        }

        if (!executorsToUnassign.isEmpty()) {
            LOG.info("Unassigning {} improperly placed executors.", executorsToUnassign.size());
            cluster.unassign(topology.getId(), executorsToUnassign);
        }
    }

    private String getTargetHost(String component, Map<String, String> placement) {
        // First try exact match
        if (placement.containsKey(component)) {
            return placement.get(component);
        }

        // If component is numeric (e.g. "0"), try "component_0"
        if (component.matches("\\d+")) {
            String key = "component_" + component;
            if (placement.containsKey(key)) {
                return placement.get(key);
            }
        } 
        // Or if placement key is numeric (e.g. "0") and component is "component_0"
        else if (component.startsWith("component_")) {
             String numericPart = component.substring("component_".length());
             if (placement.containsKey(numericPart)) {
                 return placement.get(numericPart);
             }
        }

        return null;
    }

    private void groupExecutorsByHost(Map<String, List<ExecutorDetails>> needs,
                                      Map<String, String> placement,
                                      Map<String, List<ExecutorDetails>> hostToExecs,
                                      Map<String, List<String>> hostToComponents) {
        for (Map.Entry<String, String> rule : placement.entrySet()) {
            String component = rule.getKey();
            String hostOrId = rule.getValue();

            List<ExecutorDetails> execs = needs.get(component);
            if (execs == null || execs.isEmpty()) {
                continue; // component not in this topology or already scheduled
            }

            hostToExecs.computeIfAbsent(hostOrId, k -> new ArrayList<>()).addAll(execs);
            hostToComponents.computeIfAbsent(hostOrId, k -> new ArrayList<>()).add(component);
        }
    }

    private Set<WorkerSlot> assignExecutorsFromCsv(Cluster cluster, TopologyDetails topology,
                                                   Map<String, List<ExecutorDetails>> hostToExecs,
                                                   Map<String, List<String>> hostToComponents,
                                                   Map<String, List<ExecutorDetails>> needs,
                                                   Map<String, String> hostIdToVmName) {
        Set<WorkerSlot> usedSlots = new LinkedHashSet<>();

        for (Map.Entry<String, List<ExecutorDetails>> entry : hostToExecs.entrySet()) {
            String hostOrId = entry.getKey();
            List<ExecutorDetails> execsForHost = entry.getValue();

            SupervisorDetails sup = resolveSupervisor(cluster, hostOrId, hostIdToVmName);
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

            // Only remove components from needs when assignment succeeds.
            for (String component : hostToComponents.getOrDefault(hostOrId, Collections.emptyList())) {
                needs.remove(component);
            }

            LOG.info("Packed {} executors into ONE worker on host={} port={}",
                    execsForHost.size(), sup.getHost(), chosen.getPort());
        }
        return usedSlots;
    }

    private void assignRemainingExecutors(Cluster cluster, TopologyDetails topology,
                                          Map<String, List<ExecutorDetails>> needs,
                                          Set<WorkerSlot> usedSlots) {
        // Get all available slots from cluster, excluding those already assigned in step 2
        List<WorkerSlot> availableSlots = new ArrayList<>();
        for (WorkerSlot slot : cluster.getAvailableSlots()) {
            if (!usedSlots.contains(slot)) {
                availableSlots.add(slot);
            }
        }

        // If nothing was assigned yet, also consider existing assignment slots
        if (availableSlots.isEmpty() && !usedSlots.isEmpty()) {
            LOG.warn("No more free slots available. {} components may remain unscheduled.", needs.size());
        }

        int slotIdx = 0;
        for (Map.Entry<String, List<ExecutorDetails>> rem : needs.entrySet()) {
            if (availableSlots.isEmpty()) {
                LOG.warn("No slots available for component '{}'. Remaining: {}",
                        rem.getKey(), needs.keySet());
                break;
            }

            // Simple distribution: pick a slot and use it.
            // Using modulo logic as original code did, but safer logic could be just remove(0)
            WorkerSlot target = availableSlots.get(slotIdx % availableSlots.size());
            cluster.assign(target, topology.getId(), rem.getValue());
            usedSlots.add(target);
            availableSlots.remove(slotIdx % availableSlots.size());

            LOG.info("Fallback packed component '{}' into available worker nodeId={} port={}",
                    rem.getKey(), target.getNodeId(), target.getPort());

            slotIdx++;
        }
    }

    @Override
    public Map<String, Object> config() {
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
            if (!Files.isReadable(p)) {
                LOG.warn("Placement CSV is not readable: {}", file);
                return out;
            }

            try (FileReader fileReader = new FileReader(file);
                 CSVReader reader = new CSVReader(fileReader)) {
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

                    // If component is a numeric ID, convert to "component_X" format
                    String componentKey = component;
                    if (component.matches("\\d+")) {
                        componentKey = "component_" + component;
                    }

                    out.put(componentKey, hostOrId);
                }
            }

            LOG.info("Loaded {} placement rules from CSV {}", out.size(), file);
        } catch (Exception ex) {
            LOG.error("Error reading placement CSV: " + file, ex);
        }

        return out;
    }

    private SupervisorDetails resolveSupervisor(Cluster cluster, String hostOrId, Map<String, String> hostIdToVmName) {
        // 0) Check if hostOrId is in the mapping (it's a node ID from placement CSV)
        String vmName = hostIdToVmName.get(hostOrId);
        if (vmName != null && !vmName.isBlank()) {
            LOG.debug("Mapping '{}' (node ID) to VM name '{}'", hostOrId, vmName);
            hostOrId = vmName;
        }

        // 1) Exact match by Supervisor ID
        SupervisorDetails byId = cluster.getSupervisorById(hostOrId);
        if (byId != null) {
            LOG.debug("Resolved '{}' to supervisor by ID: {} (host={})", hostOrId, byId.getId(), byId.getHost());
            return byId;
        }

        // 2) Exact match by host name
        for (SupervisorDetails s : cluster.getSupervisors().values()) {
            if (hostOrId.equals(s.getHost())) {
                LOG.debug("Resolved '{}' to supervisor by exact hostname match: {} (id={})", hostOrId, s.getHost(), s.getId());
                return s;
            }
        }

        // 3) Soft match (vm1 vs vm1.domain, etc.)
        for (SupervisorDetails s : cluster.getSupervisors().values()) {
            String h = s.getHost();
            if (h != null) {
                String hLower = h.toLowerCase(Locale.ROOT);
                String keyLower = hostOrId.toLowerCase(Locale.ROOT);
                // Match short host to FQDN or vice versa (e.g., "vm1" <-> "vm1.domain")
                if (hLower.startsWith(keyLower + ".") || keyLower.startsWith(hLower + ".")) {
                    LOG.info("Resolved '{}' to supervisor by fuzzy match: {} (id={})", hostOrId, s.getHost(), s.getId());
                    return s;
                }
            }
        }

        LOG.warn("Could not resolve '{}' to any supervisor. Available supervisors:", hostOrId);
        for (SupervisorDetails s : cluster.getSupervisors().values()) {
            LOG.warn("  - ID: {}, Host: {}", s.getId(), s.getHost());
        }

        return null;
    }

    /**
     * Log all available supervisors for debugging VM identification issues
     */
    private void logAvailableSupervisors(Cluster cluster) {
        Map<String, SupervisorDetails> supervisors = cluster.getSupervisors();
        if (supervisors.isEmpty()) {
            LOG.warn("No supervisors registered with Nimbus!");
            return;
        }

        LOG.info("=== Available Supervisors ({}) ===", supervisors.size());
        for (SupervisorDetails s : supervisors.values()) {
            LOG.info("  Supervisor ID: {}", s.getId());
            LOG.info("    Hostname: {}", s.getHost());
            LOG.info("    Total Slots: {}, Available: {}",
                s.getAllPorts().size(),
                cluster.getAvailableSlots(s).size());
        }
        LOG.info("=================================");
    }

    /**
     * Load host ID to VM name mapping from a separate CSV file
     */
    private Map<String, String> loadHostMapping(String placementCsvFile) {
        Map<String, String> mapping = new HashMap<>();
        Path mappingFile = findMappingFile(placementCsvFile);

        if (mappingFile == null) {
            return mapping;
        }

        try (FileReader fileReader = new FileReader(mappingFile.toFile());
             CSVReader reader = new CSVReader(fileReader)) {
            String[] row;
            boolean first = true;

            while ((row = reader.readNext()) != null) {
                if (row.length < 2) continue;

                if (first) {
                    first = false;
                    if ("host".equalsIgnoreCase(row[0].trim())) {
                        continue;
                    }
                }

                String hostId = row[0] == null ? "" : row[0].trim();
                String vmName = row[1] == null ? "" : row[1].trim();

                if (hostId.isEmpty() || vmName.isEmpty()) continue;
                if (hostId.startsWith("#")) continue; // allow comments

                mapping.put(hostId, vmName);
                LOG.debug("Loaded mapping: host {} -> VM {}", hostId, vmName);
            }

            LOG.info("Loaded {} host-to-VM mappings from {}", mapping.size(), mappingFile);
        } catch (Exception ex) {
            LOG.debug("Could not load host mapping file {}: {}", mappingFile, ex.getMessage());
        }

        return mapping;
    }

    private Path findMappingFile(String placementCsvFile) {
        // 1. Try deriving from placement file name: placement.csv -> placement_mapping.csv
        String mappingFileName;
        if (placementCsvFile.endsWith(".csv")) {
            mappingFileName = placementCsvFile.substring(0, placementCsvFile.length() - 4) + "_mapping.csv";
        } else {
            mappingFileName = placementCsvFile + "_mapping.csv";
        }

        Path p = Path.of(mappingFileName);
        if (Files.exists(p) && Files.isReadable(p)) {
            return p;
        }

        // 2. Fallbacks in the same directory
        Path placementPath = Path.of(placementCsvFile);
        Path parent = placementPath.getParent();
        if (parent == null || !Files.isDirectory(parent)) {
            LOG.debug("Host mapping file does not exist: {} (optional)", mappingFileName);
            return null;
        }

        // 2a. Check for generic "mapping.csv"
        Path genericMapping = parent.resolve("mapping.csv");
        if (Files.exists(genericMapping) && Files.isReadable(genericMapping)) {
            LOG.info("Using generic mapping file: {}", genericMapping);
            return genericMapping;
        }

        // 2b. Auto-detect a single *_mapping.csv
        List<Path> candidates = new ArrayList<>();
        try (var stream = Files.list(parent)) {
            stream
                .filter(Files::isRegularFile)
                .filter(path -> path.getFileName().toString().endsWith("_mapping.csv"))
                .forEach(candidates::add);
        } catch (Exception e) {
            LOG.warn("Error scanning directory for mapping files: {}", e.getMessage());
        }

        if (candidates.size() == 1) {
            Path auto = candidates.get(0);
            LOG.info("Using auto-detected mapping file: {}", auto);
            return auto;
        }

        if (candidates.size() > 1) {
            LOG.warn("Multiple *_mapping.csv files found in {}. Expected one. Candidates: {}",
                    parent, candidates);
        } else {
            LOG.debug("Host mapping file does not exist: {} (optional)", mappingFileName);
        }

        return null;
    }
}