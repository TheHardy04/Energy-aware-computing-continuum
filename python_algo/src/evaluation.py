
from dataclasses import dataclass
from typing import Dict, Any, List, Set, Tuple
import networkx as nx

from src.placementAlgo import PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph

# Constants from Code_unified_benchmark_vf.py
P_STATIC = 200
P_CPU_UNIT = 5

@dataclass
class EvaluationMetrics:
    """Dataclass to hold evaluation metrics for a placement result.
    Attributes:
        total_energy: Total energy consumption of the placement (node + link)
        energy_node: Energy consumed by active hosts based on CPU usage
        energy_link: Energy consumed by network links based on bandwidth and latency
        avg_latency: Average latency across all service edges
        worst_latency: Maximum latency among all service edges
        total_latency: Sum of latencies across all service edges
        active_hosts_count: Number of hosts that have at least one component placed on them
        host_cpu_usage: Dictionary mapping host_id to total CPU used on that host
        host_ram_usage: Dictionary mapping host_id to total RAM used on that host
        violations: List of any constraint violations detected during evaluation (e.g., capacity overflows, missing paths)
    """
    total_energy: float
    energy_node: float
    energy_link: float
    avg_latency: float 
    worst_latency: float
    total_latency: float
    active_hosts_count: int
    host_cpu_usage: Dict[int, int]
    host_ram_usage: Dict[int, int]
    violations: List[str]


class Evaluator:
    """Evaluator for placement results, calculating energy and latency metrics based on the infrastructure and application graphs."""
    
    @staticmethod
    def evaluate(infra: NetworkGraph, app: ServiceGraph, placement: PlacementResult, verbose=False) -> EvaluationMetrics:
        """
        Evaluate the given placement result against the infrastructure and application constraints.
        Args:
            infra (NetworkGraph): The infrastructure graph with host and link properties.
            app (ServiceGraph): The application graph with component and service link properties.
            placement (PlacementResult): The result of the placement algorithm to evaluate.
            verbose (bool): Whether to print detailed evaluation metrics.
        Returns:
            EvaluationMetrics: A dataclass containing energy, latency, and violation metrics.
        """
        # 1. Calculate Host Resource Usage
        host_cpu_used: Dict[int, int] = {h: 0 for h in infra.G.nodes}
        host_ram_used: Dict[int, int] = {h: 0 for h in infra.G.nodes}
        active_hosts: Set[int] = set()
        violations: List[str] = []
        
        # Mapping: service_id -> host_id
        mapping = placement.mapping

        for svc_id, host_id in mapping.items():
            if host_id not in infra.G.nodes:
                violations.append(f"Component {svc_id} mapped to non-existent host {host_id}")
                continue
                
            comp_data = app.G.nodes[svc_id]
            cpu_req = comp_data.get('cpu', 0)
            ram_req = comp_data.get('ram', 0)
            
            host_cpu_used[host_id] += cpu_req
            host_ram_used[host_id] += ram_req
            active_hosts.add(host_id)

        # Check capacity constraints
        for h in active_hosts:
            host_data = infra.G.nodes[h]
            host_cpu_cap = host_data.get('cpu', 0) or 0
            host_ram_cap = host_data.get('ram', 0) or 0
            
            if host_cpu_used[h] > host_cpu_cap:
                violations.append(f"Host {h} CPU overflow: {host_cpu_used[h]} > {host_cpu_cap}")
            if host_ram_used[h] > host_ram_cap:
                 violations.append(f"Host {h} RAM overflow: {host_ram_used[h]} > {host_ram_cap}")
        
        # 2. Calculate Node Energy
        # energy_node = sum(P_STATIC + cu[h] * P_CPU_UNIT for h in active_hosts)
        energy_node = sum(P_STATIC + host_cpu_used[h] * P_CPU_UNIT for h in active_hosts)

        # 3. Calculate Link Energy & Latency
        # energy_link = sum(flow * arc_latency^2) for all physical links used
        energy_link = 0.0
        total_latency = 0.0
        max_latency = 0.0
        
        # Iterate over all service links
        for u, v, data in app.G.edges(data=True):
            flow = data.get('bandwidth', 0)
            
            host_u = mapping.get(u)
            host_v = mapping.get(v)
            
            if host_u is None or host_v is None:
                # Should have been caught by mapping check, but just in case
                continue

            # Check if Co-located
            if host_u == host_v:
                # 0 latency, 0 link energy
                continue

            # Get the path for this service link
            path_nodes = placement.paths.get((u, v))

            if not path_nodes:
                violations.append(f"No path found for service link {u}->{v} between hosts {host_u} and {host_v}")
                continue
            
            # Verify path endpoints match placement
            if path_nodes[0] != host_u or path_nodes[-1] != host_v:
                 violations.append(f"Path endpoints {path_nodes[0]}->{path_nodes[-1]} do not match placement {host_u}->{host_v}")

            current_path_latency = 0.0
            
            if len(path_nodes) > 1:
                # Traverse the path and sum up latencies and energy
                for i in range(len(path_nodes) - 1):
                    n1 = path_nodes[i]
                    n2 = path_nodes[i+1]
                    
                    if infra.G.has_edge(n1, n2):
                        edge_data_infra = infra.G.edges[n1, n2]
                        lat = edge_data_infra.get('latency', 0)
                        
                        current_path_latency += lat
                        # Energy logic: flow * lat^2
                        energy_link += flow * (lat ** 2)
                    else:
                        violations.append(f"Physical link {n1}->{n2} does not exist for service link {u}->{v}")
            
            total_latency += current_path_latency
            if current_path_latency > max_latency:
                max_latency = current_path_latency

        num_links = app.G.number_of_edges()
        avg_latency = total_latency / num_links if num_links > 0 else 0

        total_energy = energy_node + energy_link

        evaluation_result = EvaluationMetrics(
            total_energy=total_energy,
            energy_node=energy_node,
            energy_link=energy_link,
            avg_latency=avg_latency,
            worst_latency=max_latency,
            total_latency=total_latency,
            active_hosts_count=len(active_hosts),
            host_cpu_usage=host_cpu_used,
            host_ram_usage=host_ram_used,
            violations=violations
        )
        if verbose:
            Evaluator.print_metrics(evaluation_result)

        return evaluation_result
    
    @staticmethod
    def print_metrics(metrics: EvaluationMetrics):
        """Utility method to print evaluation metrics in a readable format.
        Args:
            metrics (EvaluationMetrics): The evaluation metrics to print.
        """
        print("Evaluation Metrics:")
        print(f"  Total Energy: {metrics.total_energy:.2f}")
        print(f"    - Node Energy: {metrics.energy_node:.2f}")
        print(f"    - Link Energy: {metrics.energy_link:.2f}")
        print(f"  Average Latency: {metrics.avg_latency:.2f}")
        print(f"  Worst Latency: {metrics.worst_latency:.2f}")
        print(f"  Total Latency: {metrics.total_latency:.2f}")
        print(f"  Active Hosts: {metrics.active_hosts_count}")
        print("  Host CPU Usage:")
        for h, cpu in metrics.host_cpu_usage.items():
            print(f"    Host {h}: CPU used = {cpu}")
        print("  Host RAM Usage:")
        for h, ram in metrics.host_ram_usage.items():
            print(f"    Host {h}: RAM used = {ram}")
        if metrics.violations:
            print("  Violations:")
            for v in metrics.violations:
                print(f"    - {v}")
        else:
            print("  No violations detected.")
