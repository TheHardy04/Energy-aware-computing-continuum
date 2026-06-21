from typing import Dict, Any, Tuple, List


def host_resources_snapshot(network_graph) -> Dict[int, Dict[str, Any]]:
    """
    Create a snapshot of host resources (cpu, ram) from the network graph.
    """
    resources: Dict[int, Dict[str, Any]] = {}
    for n, d in network_graph.G.nodes(data=True):
        resources[n] = {
            'cpu_total': int(d.get('cpu') or 0),
            'ram_total': int(d.get('ram') or 0),
            'cpu_used': 0,
            'ram_used': 0,
        }
    return resources

def edge_ressources_snapshot(network_graph) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Create a snapshot of edge resources (bandwidth, latency) from the network graph.
    """
    resources: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for u, v, d in network_graph.G.edges(data=True):
        resources[(u, v)] = {
            'bandwidth_total': int(d.get('bandwidth') or 0),
            'latency': int(d.get('latency') or 0),
            'bandwidth_used': 0,
        }
    return resources


def can_host(resources: Dict[int, Dict[str, Any]], node: int, cpu: int, ram: int) -> bool:
    """
    Check if the given host node can accommodate the cpu/ram request.
    """
    r = resources[node]
    return r['cpu_used'] + cpu <= r['cpu_total'] and r['ram_used'] + ram <= r['ram_total']


def allocate_on_host(resources: Dict[int, Dict[str, Any]], node: int, cpu: int, ram: int) -> None:
    """
    Consume cpu/ram on the given host node.
    """
    resources[node]['cpu_used'] += cpu
    resources[node]['ram_used'] += ram

def edge_capacity_ok(edge_resources: Dict[Tuple[int, int], Dict[str, Any]], path: List[int], bandwidth: int) -> bool:
    """
    Check if all edges along the path can accommodate the bandwidth request.
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        r = edge_resources.get((u, v))
        if r is None or r['bandwidth_used'] + bandwidth > r['bandwidth_total']:
            return False
    return True

def allocate_on_edges(edge_resources: Dict[Tuple[int, int], Dict[str, Any]], path: List[int], bandwidth: int) -> None:
    """
    Consume bandwidth on all edges along the path.
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_resources[(u, v)]['bandwidth_used'] += bandwidth
