import logging
import copy
from typing import Dict, List, Tuple, Any, Optional
from src.placementAlgo import PlacementResult
from src.serviceGraph import ServiceGraph
from src.networkGraph import NetworkGraph


class MappingValidator:
    """
    A validator class to check the correctness of placement results against
    network and service constraints. This acts as a runtime verification tool.
    It collects all errors found during validation instead of stopping at the first failure.
    """

    def __init__(self, network_graph: NetworkGraph, service_graph: ServiceGraph, final_placement: PlacementResult, logger: Optional[logging.Logger] = None):
        self.network_graph = network_graph
        self.service_graph = service_graph
        self.final_placement = final_placement
        self.errors: List[str] = []
        self.logger = logger

    def _assert(self, condition: bool, message: str):
        """Helper to collect errors instead of raising exceptions immediately."""
        if not condition:
            self.errors.append(message)
            if self.logger:
                self.logger.error(message)

    def validate(self) -> bool:
        """
        Runs all validation checks.
        Returns True if all checks pass, False otherwise.
        """
        if self.logger:
            self.logger.info("Starting Mapping Validation...")
        self.errors = [] # Reset errors

        try:
            self.validate_structure()
            # If structure is invalid, other tests might crash, so checking basic integrity first
            if not self.errors:
                self.validate_placement_integrity()
                self.validate_host_resources()
                self.validate_routing_constraints()
                self.validate_cycles()
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Unexpected error during validation: {e}")
            self.errors.append(f"Crash during validation: {str(e)}")

        if not self.errors:
            if self.logger:
                self.logger.info("✅ All Mapping Validation Tests Passed!")
            return True
        else:
            if self.logger:
                self.logger.error(f"❌ Mapping Validation Failed with {len(self.errors)} errors:")
                for err in self.errors:
                    self.logger.error(f"  - {err}")
            return False

    def validate_structure(self):
        """Checks if graphs are non-empty."""
        self._assert(self.network_graph.G.number_of_nodes() > 0, "Network graph is empty")
        self._assert(self.network_graph.G.number_of_edges() > 0, "Network graph has no edges")
        self._assert(self.service_graph.G.number_of_nodes() > 0, "Service graph is empty")
        self._assert(self.service_graph.G.number_of_edges() > 0, "Service graph has no edges")
        # Only log success if no errors were added in this step
        if not self.errors:
            if self.logger:
                self.logger.info("Structure check passed.")

    def validate_placement_integrity(self):
        """Checks if all components are placed on valid hosts."""
        validation_passed = True
        
        # Check all placed components
        for comp, host in self.final_placement.mapping.items():
            if host not in self.network_graph.G.nodes():
                self._assert(False, f"Component {comp} placed on invalid host {host}")
                validation_passed = False
            if comp not in self.service_graph.G.nodes():
                self._assert(False, f"Invalid component {comp} in placement mapping")
                validation_passed = False
        
        # Check if all service nodes are placed
        for node in self.service_graph.G.nodes():
             if node not in self.final_placement.mapping:
                 self._assert(False, f"Service node {node} is missing from placement")
                 validation_passed = False
        
        if validation_passed:
            if self.logger:
                self.logger.info("Placement integrity check passed.")

    def validate_host_resources(self):
        """Checks if host CPU and RAM capacities are respected."""
        # Initialize used resources
        # We need to be careful to use the same node identifiers
        used_resources = {n: {'cpu': 0, 'ram': 0} for n in self.network_graph.G.nodes()}
        
        # Sum up resources used by components
        for comp, host in self.final_placement.mapping.items():
            if host not in used_resources:
                continue # Error already caught in integrity check
            
            comp_data = self.service_graph.G.nodes[comp]
            used_resources[host]['cpu'] += comp_data.get('cpu', 0)
            used_resources[host]['ram'] += comp_data.get('ram', 0)

        # Compare against capacities
        resource_issues = 0
        for host, used in used_resources.items():
            host_data = self.network_graph.G.nodes[host]
            total_cpu = host_data.get('cpu', 0)
            total_ram = host_data.get('ram', 0)
            
            if used['cpu'] > total_cpu:
                self._assert(False, f"Host {host} CPU overcommit: Used {used['cpu']} > Total {total_cpu}")
                resource_issues += 1
            
            if used['ram'] > total_ram:
                self._assert(False, f"Host {host} RAM overcommit: Used {used['ram']} > Total {total_ram}")
                resource_issues += 1
        
        if resource_issues == 0:
            if self.logger:
                self.logger.info("Host resource check passed.")

    def validate_routing_constraints(self):
        """Checks bandwidth, latency, and path continuity constraints."""
        paths = self.final_placement.paths
        
        # Deep copy to track bandwidth consumption without modifying original graph
        # This assumes the network graph structure allows deepcopy
        try:
            network_check_graph = copy.deepcopy(self.network_graph.G)
        except Exception:
            # Fallback if deepcopy fails (e.g. unpicklable objects)
            if self.logger:
                self.logger.warning("Could not deepcopy network graph for bandwidth check. skipping bandwidth accumulation.")
            network_check_graph = self.network_graph.G

        routing_issues = 0
        
        for (u, v), path_nodes in paths.items():
            # Get service edge requirements
            # Ensure u, v are in correct type (int vs str) if necessary
            edge_data = self.service_graph.G.get_edge_data(u, v)
            if edge_data is None:
                # It's possible the paths keys are directed but service graph is undirected or vice versa
                # Try reverse if not found
                edge_data = self.service_graph.G.get_edge_data(v, u)

            if edge_data is None:
                self._assert(False, f"Placement contains path for non-existent service edge {u}->{v}")
                routing_issues += 1
                continue

            bw_req = edge_data.get('bandwidth', 0)
            lat_limit = edge_data.get('latency', float('inf'))

            # Check path endpoints
            src_host = self.final_placement.mapping.get(u)
            dst_host = self.final_placement.mapping.get(v)
            
            if not path_nodes:
                 # If path is empty, src and dst MUST be the same (colocation)
                 if src_host != dst_host:
                     self._assert(False, f"Empty path for {u}->{v} but misplaced hosts: {src_host} != {dst_host}")
                     routing_issues += 1
                 continue

            if path_nodes[0] != src_host:
                self._assert(False, f"Path for {u}->{v} starts at {path_nodes[0]}, expected {src_host}")
                routing_issues += 1
            
            if path_nodes[-1] != dst_host:
                self._assert(False, f"Path for {u}->{v} ends at {path_nodes[-1]}, expected {dst_host}")
                routing_issues += 1

            # Check links in path
            total_latency = 0
            for i in range(len(path_nodes)-1):
                h1, h2 = path_nodes[i], path_nodes[i+1]
                
                if not network_check_graph.has_edge(h1, h2):
                    self._assert(False, f"Path for {u}->{v} uses non-existent link {h1}->{h2}")
                    routing_issues += 1
                    continue

                link_attr = network_check_graph.get_edge_data(h1, h2)
                total_latency += link_attr.get('latency', 0)
                
                # Check bandwidth availability
                current_bw = link_attr.get('bandwidth', 0)
                if current_bw < bw_req:
                    # Only report this once per link preferrably, but for now reporting per flow is fine
                    self._assert(False, 
                                f"Link {h1}->{h2} oversubscribed by {u}->{v}. "
                                f"Required: {bw_req}, Available: {current_bw}")
                    routing_issues += 1
                else:
                    # Deduct bandwidth for subsequent checks
                    link_attr['bandwidth'] = current_bw - bw_req
            
            if total_latency > lat_limit:
                self._assert(False, 
                            f"Path for {u}->{v} exceeds latency limit. "
                            f"Actual: {total_latency} > Limit: {lat_limit}")
                routing_issues += 1

        if routing_issues == 0:
            if self.logger:
                self.logger.info("Routing constraints check passed.")

    def validate_cycles(self):
        """Checks for cycles in routing paths."""
        paths = self.final_placement.paths
        cycle_issues = 0
        for (u, v), path in paths.items():
            if len(path) != len(set(path)):
                self._assert(False, f"Cycle detected in path for service edge {u}->{v}: {path}")
                cycle_issues += 1
        
        if cycle_issues == 0:
            if self.logger:
                self.logger.info("Cycle check passed.")


# For backward compatibility or direct usage
class MappingUnitTest:
    @staticmethod
    def run_tests(network_graph: NetworkGraph, service_graph: ServiceGraph, final_placement: PlacementResult, logger: Optional[logging.Logger] = None) -> bool:
        """Convenience method to run all mapping validation tests.
        Returns True if all tests pass, False otherwise.
        """
        validator = MappingValidator(network_graph, service_graph, final_placement, logger=logger)
        return validator.validate()
