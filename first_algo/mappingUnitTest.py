from src.base import PlacementResult
from src.serviceGraph import *
from src.networkGraph import *
import copy

class MappingUnitTest:
    """
    A collection of unit tests to validate the correctness of the mapping results.
    """

    @staticmethod
    def run_tests(network_graph : NetworkGraph, service_graph : ServiceGraph, final_placement : PlacementResult):
        """
        Docstring for run_tests
        
        :param network_graph: The network graph representing the infrastructure
        :type network_graph: NetworkGraph
        :param service_graph: The service graph representing the application
        :type service_graph: ServiceGraph
        :param final_placement: The final placement result to be validated
        :type final_placement: PlacementResult
        """
        print("Running Mapping Unit Tests...")
        MappingUnitTest.validate_network_structure(network_graph)
        MappingUnitTest.validate_service_structure(service_graph)
        MappingUnitTest.validate_final_placement(final_placement, network_graph, service_graph)
        MappingUnitTest.validate_host_resources(network_graph, service_graph, final_placement)
        MappingUnitTest.validate_edge_routing_constraints(network_graph, service_graph, final_placement)
        MappingUnitTest.test_no_cycles_in_routing(final_placement)

        print("All Mapping Unit Tests Passed!")

    @staticmethod
    def validate_network_structure(network_graph: NetworkGraph):
        # Test 1: Validate the network graph structure
        assert network_graph.G.number_of_nodes() > 0, "Network graph is empty"
        assert network_graph.G.number_of_edges() > 0, "Network graph has no edges"
        print("Network structure tests passed.")

    @staticmethod
    def validate_service_structure(service_graph: ServiceGraph):
        # Test 2: Validate the service graph structure
        assert service_graph.G.number_of_nodes() > 0, "Service graph is empty"
        assert service_graph.G.number_of_edges() > 0, "Service graph has no edges"
        print("Service structure tests passed.")

    @staticmethod
    def validate_final_placement(final_placement: PlacementResult, network_graph: NetworkGraph, service_graph: ServiceGraph):
        # Test 3: Validate the final placement
        for comp, host in final_placement.mapping.items():
            assert host in network_graph.G.nodes(), f"Component {comp} placed on invalid host {host}"
            assert comp in service_graph.G.nodes(), f"Invalid component {comp} in placement mapping"
        print("Final placement mapping tests passed.")

    @staticmethod
    def validate_host_resources(network_graph: NetworkGraph, service_graph: ServiceGraph, final_placement: PlacementResult):
        # Test 4: Validate host resource constraints
        host_resources = {n: {'cpu': d.get('cpu'), 'ram': d.get('ram')} for n, d in network_graph.G.nodes(data=True)}
        used_resources = {n: {'cpu': 0, 'ram': 0} for n in network_graph.G.nodes()}
        for comp, host in final_placement.mapping.items():
            comp_data = service_graph.G.nodes[comp]
            used_resources[host]['cpu'] += comp_data.get('cpu', 0)
            used_resources[host]['ram'] += comp_data.get('ram', 0)

        for host, used in used_resources.items():
            total = host_resources.get(host, {'cpu': 0, 'ram': 0})
            assert used['cpu'] <= total['cpu'], f"Host {host} CPU overcommit: {used['cpu']} > {total['cpu']}"
            assert used['ram'] <= total['ram'], f"Host {host} RAM overcommit: {used['ram']} > {total['ram']}"
        print("Host resource constraint tests passed.")

    @staticmethod
    def validate_edge_routing_constraints(network_graph: NetworkGraph, service_graph: ServiceGraph, final_placement: PlacementResult):
        # Test 5: Validate edge routing constraints
        path = final_placement.paths # Dict[Tuple[int, int], List[int]]
        reconstructed = copy.deepcopy(network_graph)

        for (u, v), p in path.items():
            edge_data = service_graph.G.get_edge_data(u, v)
            if edge_data is None:
                raise AssertionError(f"No edge data for service edge {u}->{v}")
            bw_req = edge_data.get('bandwidth', 0)
            lat_limit = edge_data.get('latency', float('inf'))

            # Validate path starts and ends correctly
            src_host = final_placement.mapping[u]
            dst_host = final_placement.mapping[v]
            assert p[0] == src_host, f"Path for edge {u}->{v} does not start at source host {src_host}"
            assert p[-1] == dst_host, f"Path for edge {u}->{v} does not end at destination host {dst_host}"

            # Validate path edges exist and accumulate latency
            total_latency = 0
            for i in range(len(p)-1):
                if not reconstructed.G.has_edge(p[i], p[i+1]):
                    raise AssertionError(f"Path for edge {u}->{v} contains invalid edge {p[i]}->{p[i+1]}")
                edge_attr = reconstructed.G.get_edge_data(p[i], p[i+1])
                total_latency += edge_attr.get('latency', 0)
                assert total_latency <= lat_limit, f"Path for edge {u}->{v} exceeds latency limit: {total_latency} > {lat_limit}"
                
                assert "bandwidth" in edge_attr, f"Edge {p[i]}->{p[i+1]} missing bandwidth attribute"
                # Consume bandwidth
                if 'bandwidth' in edge_attr:
                    if edge_attr['bandwidth'] < bw_req:
                        raise AssertionError(f"Edge {p[i]}->{p[i+1]} cannot accommodate bandwidth {bw_req} for service edge {u}->{v}")
                    edge_attr['bandwidth'] -= bw_req

        print("Edge routing constraint tests passed.")

    @staticmethod
    def test_no_cycles_in_routing(final_placement: PlacementResult):
        # Test 6: Ensure no cycles in routing paths
        routing = final_placement.meta.get('routing', {})
        for (u, v), info in routing.items():
            path = info.get('path', [])
            assert len(path) == len(set(path)), f"Cycle detected in routing path for edge {u}->{v}"
        print("All cycle detection tests passed.")