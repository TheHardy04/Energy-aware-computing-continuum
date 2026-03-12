import copy
from typing import Dict, Tuple, List
import networkx as nx

from src.placementAlgo import PlacementAlgo, PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph
from src.utils import host_resources_snapshot, edge_ressources_snapshot, can_host, allocate_on_host, edge_capacity_ok, allocate_on_edges


class GreedyFirstFit(PlacementAlgo):
    """First Fit Decreasing placement with local backtracking.

    Components are processed in decreasing CPU order. Each component is tentatively
    assigned to the first host that satisfies host capacities, then its service links
    toward already placed neighbours are validated immediately. If one of those link
    constraints fails, the tentative assignment is rolled back and the next host is tried.
    """

    def place(self, service_graph: ServiceGraph, network_graph: NetworkGraph, **kwargs) -> PlacementResult:
        SG, NG = service_graph.G, network_graph.G

        res = host_resources_snapshot(network_graph)
        edge_res = edge_ressources_snapshot(network_graph)

        mapping: Dict[int, int] = {}
        paths: Dict[Tuple[int, int], List[int]] = {}
        hosts_list = list(NG.nodes())

        def path_latency(path: List[int]) -> float:
            return sum(
                float(NG.get_edge_data(path[i], path[i + 1], default={}).get('latency', 0))
                for i in range(len(path) - 1)
            )

        def build_dz_map() -> Dict[int, int]:
            dz_map: Dict[int, int] = {}
            dz_data = service_graph.metadata.get('component.DZ')
            if not isinstance(dz_data, list):
                return dz_map

            for i in range(0, len(dz_data), 2):
                if i + 1 >= len(dz_data):
                    break
                comp_id = dz_data[i]
                host_id = dz_data[i + 1]
                if comp_id not in SG.nodes():
                    continue
                if host_id not in NG.nodes():
                    raise ValueError(f'DZ_host_not_found_{host_id}')
                dz_map[comp_id] = host_id
            return dz_map

        def collect_incident_edges(comp: int) -> List[Tuple[int, int]]:
            incident: List[Tuple[int, int]] = []
            for pred in SG.predecessors(comp):
                if pred in mapping:
                    incident.append((pred, comp))
            for succ in SG.successors(comp):
                if succ in mapping:
                    incident.append((comp, succ))
            return incident

        def try_host(comp: int, host: int) -> Tuple[bool, Dict[Tuple[int, int], List[int]], str]:
            cpu_req = int(SG.nodes[comp].get('cpu') or 0)
            ram_req = int(SG.nodes[comp].get('ram') or 0)

            if not can_host(res, host, cpu_req, ram_req):
                return False, {}, f'capacity_fail_{comp}_on_{host}'

            trial_edge_res = copy.deepcopy(edge_res)
            candidate_paths: Dict[Tuple[int, int], List[int]] = {}

            for edge in collect_incident_edges(comp):
                src_comp, dst_comp = edge
                src_host = host if src_comp == comp else mapping[src_comp]
                dst_host = host if dst_comp == comp else mapping[dst_comp]
                bw_req = int(SG.edges[edge].get('bandwidth') or 0)
                max_latency = float(SG.edges[edge].get('latency') or float('inf'))

                if src_host == dst_host:
                    candidate_paths[edge] = [src_host]
                    continue

                try:
                    candidate_path = nx.shortest_path(NG, source=src_host, target=dst_host, weight='latency')
                except nx.NetworkXNoPath:
                    return False, {}, f'no_path_{src_comp}_{dst_comp}'

                if path_latency(candidate_path) > max_latency:
                    return False, {}, f'latency_fail_{src_comp}_{dst_comp}'

                if not edge_capacity_ok(trial_edge_res, candidate_path, bw_req):
                    return False, {}, f'link_fail_{src_comp}_{dst_comp}'

                allocate_on_edges(trial_edge_res, candidate_path, bw_req)
                candidate_paths[edge] = candidate_path

            return True, candidate_paths, 'ok'

        try:
            dz_map = build_dz_map()
        except ValueError as exc:
            return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': str(exc)})

        dz_components = sorted(
            [comp for comp in SG.nodes() if comp in dz_map],
            key=lambda comp: (-int(SG.nodes[comp].get('cpu') or 0), comp)
        )
        free_components = sorted(
            [comp for comp in SG.nodes() if comp not in dz_map],
            key=lambda comp: (-int(SG.nodes[comp].get('cpu') or 0), comp)
        )
        components = dz_components + free_components

        for comp in components:
            cpu_req = int(SG.nodes[comp].get('cpu') or 0)
            ram_req = int(SG.nodes[comp].get('ram') or 0)
            candidate_hosts = [dz_map[comp]] if comp in dz_map else hosts_list

            placed = False
            last_reason = f'no_host_for_component_{comp}'

            for host in candidate_hosts:
                is_valid, candidate_paths, reason = try_host(comp, host)
                if not is_valid:
                    last_reason = reason
                    continue

                allocate_on_host(res, host, cpu_req, ram_req)
                mapping[comp] = host
                for edge, candidate_path in candidate_paths.items():
                    bw_req = int(SG.edges[edge].get('bandwidth') or 0)
                    allocate_on_edges(edge_res, candidate_path, bw_req)
                    paths[edge] = candidate_path
                placed = True
                break

            if not placed:
                if comp in dz_map and last_reason.startswith('capacity_fail_'):
                    last_reason = f'DZ_capacity_fail_{comp}_on_{dz_map[comp]}'
                return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': last_reason})

        return PlacementResult(
            mapping=mapping,
            paths=paths,
            meta={
                'status': 'success',
                'placement_order': components,
            },
        )
