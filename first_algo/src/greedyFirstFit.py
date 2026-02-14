from typing import Dict, Tuple, List
import networkx as nx

from src.placementAlgo import PlacementAlgo
from src.base import PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph
from src.utils import host_resources_snapshot, edge_ressources_snapshot, can_host, allocate_on_host, edge_capacity_ok, allocate_on_edges


class GreedyFirstFit(PlacementAlgo):
    """A simple baseline placement:
    - Start with the first component and place it on the first host that has enough CPU/RAM.
    - Continue the next component, look for the first host near the first one that can accommodate it verifying also the bandwidth/latency constraints and so on until all components are placed or we fail to place one.
    - Returns mapping and per-edge routing meta.
    """

    def place(self, service_graph : ServiceGraph, network_graph : NetworkGraph, start_host: int = 0) -> PlacementResult:
        SG, NG = service_graph.G, network_graph.G

        # Track host resources
        res = host_resources_snapshot(network_graph)
        # Track edge resources 
        edge_res = edge_ressources_snapshot(network_graph)

        mapping: Dict[int, int] = {}
        paths: Dict[Tuple[int, int], List[int]] = {}

        # 1) Place first component on the first host that can accommodate it starting from start_host if provided
        hosts_list = list(NG.nodes())
        if start_host in hosts_list:
            # rotate so start_host is first
            idx = hosts_list.index(start_host)
            hosts_list = hosts_list[idx:] + hosts_list[:idx]
        
        # Get first component and its requirements
        first_comp = list(SG.nodes())[0]
        cpu_req = int(SG.nodes[first_comp].get('cpu') or 0)
        ram_req = int(SG.nodes[first_comp].get('ram') or 0)

        # Try to place first component on the first host that can accommodate it
        placed = False
        for host in hosts_list:
            if can_host(res, host, cpu_req, ram_req):
                allocate_on_host(res, host, cpu_req, ram_req)
                mapping[first_comp] = host
                placed = True
                break
        if not placed:
            return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_host_for_component_{first_comp}'})
        
        # 2) Place remaining components iteratively, trying to place near already placed ones
        for comp, d in SG.nodes(data=True):
            if comp in mapping:
                continue
            cpu_req = int(d.get('cpu') or 0)
            ram_req = int(d.get('ram') or 0)
            placed = False
            # Try to place near last placed component
            last_host = mapping[list(mapping.keys())[-1]]
            # Get neighbors of last_host sorted by latency
            neighbors = sorted(NG.neighbors(last_host), key=lambda n: float(NG.edges.get((last_host, n), {}).get('latency', 0)))
            for host in neighbors:
                if can_host(res, host, cpu_req, ram_req):
                    # Check bandwidth/latency constraints from last_host to this host
                    path = nx.shortest_path(NG, source=last_host, target=host, weight='latency')
                    if edge_capacity_ok(edge_res, path, int(d.get('bandwidth') or 0)):
                        allocate_on_host(res, host, cpu_req, ram_req)
                        allocate_on_edges(edge_res, path, int(d.get('bandwidth') or 0))
                        paths[(list(mapping.keys())[-1], comp)] = path
                        mapping[comp] = host
                        placed = True
                        break
            if not placed:
                return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_host_for_component_{comp}'})
            
        return PlacementResult(mapping=mapping, paths=paths, meta={'status': 'success'})
