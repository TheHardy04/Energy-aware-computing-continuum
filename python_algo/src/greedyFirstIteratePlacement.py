from typing import Dict, Any, List, Tuple

import networkx as nx

from src.placementAlgo import PlacementAlgo
from src.base import PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph
from src.utils import host_resources_snapshot, edge_ressources_snapshot, can_host, allocate_on_host, edge_capacity_ok, allocate_on_edges


class GreedyFirstIterate(PlacementAlgo):
    """A simple baseline placement:
    - Iterate components in order (0..n-1)
    - If a component has a DZ hard constraint, place it there.
    - Otherwise, for each, pick the first host with enough CPU/RAM
    - After mapping all nodes, validate each service edge by finding a path that meets BW/latency
      using shortest path (by latency) and checking capacities.
    - Returns mapping and per-edge routing meta.
    """

    def place(self, service_graph: ServiceGraph, network_graph: NetworkGraph, **kwargs) -> PlacementResult:
        
        SG, NG = service_graph.G, network_graph.G

        # Track host resources
        res = host_resources_snapshot(network_graph)
        # Track edge resources 
        edge_res = edge_ressources_snapshot(network_graph)

        # 1) Place components
        mapping: Dict[int, int] = {}
        hosts_list = list(NG.nodes())

        # Pre-process DZ constraints
        dz_data = service_graph.metadata.get('component.DZ')
        dz_map: Dict[int, int] = {} # comp_id -> host_id

        if dz_data and isinstance(dz_data, list) and len(dz_data) >= 2:
             for i in range(0, len(dz_data), 2):
                if i+1 >= len(dz_data):
                    break
                comp_id = dz_data[i]
                host_id = dz_data[i+1]
                dz_map[comp_id] = host_id
        
        # Iterate components in order and place on first-fit host
        for comp in SG.nodes():
            d = SG.nodes[comp]
            cpu_req = int(d.get('cpu') or 0)
            ram_req = int(d.get('ram') or 0)
            
            # Check if hard constraint exists
            target_host = dz_map.get(comp)
            
            placed = False
            
            if target_host is not None:
                # Must place here
                if target_host not in NG.nodes():
                     return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'DZ_host_not_found_{target_host}'})
                
                if can_host(res, target_host, cpu_req, ram_req):
                    allocate_on_host(res, target_host, cpu_req, ram_req)
                    mapping[comp] = target_host
                    placed = True
                else:
                    return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'DZ_capacity_fail_{comp}_on_{target_host}'})
            else:
                # Try first fit on any host
                for host in hosts_list:
                    if can_host(res, host, cpu_req, ram_req):
                        allocate_on_host(res, host, cpu_req, ram_req)
                        mapping[comp] = host
                        placed = True
                        break
            
            if not placed:
                return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_host_for_component_{comp}'})

        # 2) Route edges with constraints
        # Build a latency-weighted graph for shortest paths
        H = nx.DiGraph()
        for u, v, d in NG.edges(data=True):
            lat = float(d.get('latency') or 0.0)
            H.add_edge(u, v, weight=lat)

        routing: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for u, v, d in SG.edges(data=True):
            src_host = mapping[u]
            dst_host = mapping[v]
            bw_req = int(d.get('bandwidth') or 0)
            lat_limit = int(d.get('latency') or 10**9)  # large if not provided

            try:
                path = nx.shortest_path(H, source=src_host, target=dst_host, weight='weight')
            except nx.NetworkXNoPath:
                return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_path_{u}_{v}'})

            if not edge_capacity_ok(edge_res, path, bw_req):
                return PlacementResult(mapping=mapping, paths=paths, meta={'status': 'failed', 'reason': f'constraints_{u}_{v}'})
            allocate_on_edges(edge_res, path, bw_req)

            routing[(u, v)] = {
                'path': path,
                'bandwidth': bw_req,
                'latency_limit': lat_limit,
            }

        paths = {k: v['path'] for k, v in routing.items()}
        return PlacementResult(mapping=mapping, paths=paths, meta={'status': 'success'})
