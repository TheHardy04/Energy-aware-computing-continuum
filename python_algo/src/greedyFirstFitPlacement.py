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

    def place(self, service_graph: ServiceGraph, network_graph: NetworkGraph, **kwargs) -> PlacementResult:
        SG, NG = service_graph.G, network_graph.G

        # Track host resources
        res = host_resources_snapshot(network_graph)
        # Track edge resources 
        edge_res = edge_ressources_snapshot(network_graph)

        mapping: Dict[int, int] = {}
        paths: Dict[Tuple[int, int], List[int]] = {}
        queue = []
        enqueued = set()

        # 1) Handle DZ Constraints (Hard placement)
        dz_data = service_graph.metadata.get('component.DZ')
        
        # Ensure dz_data is a list
        if dz_data is None:
            dz_data = []
        elif not isinstance(dz_data, list):
            # If it's not a list, try to handle it or ignore it safely
            dz_data = []

        # Parse DZ list [comp, host, comp, host...]
        if len(dz_data) >= 2:
             for i in range(0, len(dz_data), 2):
                if i+1 >= len(dz_data):
                    break
                comp_id = dz_data[i]
                host_id = dz_data[i+1]
                
                # Check if host exists
                if host_id not in NG.nodes():
                    return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'DZ_host_not_found_{host_id}'})
                
                # Check component exists (optional but good)
                if comp_id not in SG.nodes():
                     continue # Or fail? usually ignore if component not in current graph subset

                # Check Capacity
                cpu_req = int(SG.nodes[comp_id].get('cpu') or 0)
                ram_req = int(SG.nodes[comp_id].get('ram') or 0)

                if can_host(res, host_id, cpu_req, ram_req):
                    allocate_on_host(res, host_id, cpu_req, ram_req)
                    mapping[comp_id] = host_id
                    queue.append(comp_id)
                    enqueued.add(comp_id)
                else:
                    return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'DZ_capacity_fail_{comp_id}_on_{host_id}'})

        # 2) If Queue is empty (no DZ), start with first component on first available host
        all_nodes = list(SG.nodes())
        hosts_list = list(NG.nodes())

        if not queue and all_nodes:
            first_comp = all_nodes[0]
            cpu_req = int(SG.nodes[first_comp].get('cpu') or 0)
            ram_req = int(SG.nodes[first_comp].get('ram') or 0)
            
            placed = False
            for host in hosts_list:
                 if can_host(res, host, cpu_req, ram_req):
                    allocate_on_host(res, host, cpu_req, ram_req)
                    mapping[first_comp] = host
                    queue.append(first_comp)
                    enqueued.add(first_comp)
                    placed = True
                    break
            
            if not placed:
                return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_host_for_start_component_{first_comp}'})

        # 3) BFS / Greedy Traversal
        while len(mapping) < len(all_nodes):
            if not queue:
                # Disconnected components handling
                # Find first unplaced node
                for n in all_nodes:
                    if n not in mapping:
                        # Place it on any available host
                        cpu_req = int(SG.nodes[n].get('cpu') or 0)
                        ram_req = int(SG.nodes[n].get('ram') or 0)
                        placed_new_root = False
                        
                        for h in hosts_list:
                            if can_host(res, h, cpu_req, ram_req):
                                allocate_on_host(res, h, cpu_req, ram_req)
                                mapping[n] = h
                                queue.append(n)
                                enqueued.add(n)
                                placed_new_root = True
                                break
                        if not placed_new_root:
                             return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_host_for_disconnected_{n}'})
                        break
            
            while queue:
                curr_comp = queue.pop(0)
                curr_host = mapping[curr_comp]

                # Check all successors
                for next_comp in SG.successors(curr_comp):
                    if next_comp in mapping:
                        # Already placed; ensure (curr_comp, next_comp) path (flow) is valid?
                        # With strict BFS, usually we place edges as we discover nodes.
                        # But if next_comp was placed via another parent, we must check edge validity now.
                        if (curr_comp, next_comp) not in paths:
                            dst_host = mapping[next_comp]
                            bw_req = int(SG.edges[(curr_comp, next_comp)].get('bandwidth') or 0)
                            try:
                                p = nx.shortest_path(NG, source=curr_host, target=dst_host, weight='latency')
                                if edge_capacity_ok(edge_res, p, bw_req):
                                    allocate_on_edges(edge_res, p, bw_req)
                                    paths[(curr_comp, next_comp)] = p
                                else:
                                     return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'link_fail_{curr_comp}_{next_comp}'})
                            except nx.NetworkXNoPath:
                                 return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_path_{curr_comp}_{next_comp}'})
                        continue

                    # Attempt to place 'next_comp'
                    d = SG.nodes[next_comp]
                    cpu_req = int(d.get('cpu') or 0)
                    ram_req = int(d.get('ram') or 0)
                    
                    # Find candidates near curr_host
                    sorted_neig = sorted(
                        NG.neighbors(curr_host), 
                        key=lambda n: float(NG.edges.get((curr_host, n), {}).get('latency', 0))
                    )
                    candidates = [curr_host] + sorted_neig

                    # Identify ALL placed predecessors to validate combined constraints
                    # (One of them is curr_comp)
                    placed_preds = [p for p in SG.predecessors(next_comp) if p in mapping]

                    placed = False
                    for host in candidates:
                        if can_host(res, host, cpu_req, ram_req):
                            # Check connectivity from ALL placed predecessors
                            valid_connectivity = True
                            temp_paths = {}
                            
                            for p in placed_preds:
                                p_host = mapping[p]
                                bw = int(SG.edges[(p, next_comp)].get('bandwidth') or 0)
                                try:
                                    pth = nx.shortest_path(NG, source=p_host, target=host, weight='latency')
                                    if not edge_capacity_ok(edge_res, pth, bw):
                                        valid_connectivity = False
                                        break
                                    temp_paths[(p, next_comp)] = pth
                                except nx.NetworkXNoPath:
                                    valid_connectivity = False
                                    break
                            
                            if valid_connectivity:
                                # Commit
                                allocate_on_host(res, host, cpu_req, ram_req)
                                for k, pth in temp_paths.items():
                                    bw = int(SG.edges[k].get('bandwidth') or 0)
                                    allocate_on_edges(edge_res, pth, bw)
                                    paths[k] = pth
                                
                                mapping[next_comp] = host
                                placed = True
                                if next_comp not in enqueued:
                                    queue.append(next_comp)
                                    enqueued.add(next_comp)
                                break
                    
                    if not placed:
                        return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_host_for_component_{next_comp}'})
            
        return PlacementResult(mapping=mapping, paths=paths, meta={'status': 'success'})
