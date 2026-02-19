from typing import Dict, Tuple, List, Any

import networkx as nx
from ortools.sat.python import cp_model

from src.placementAlgo import PlacementAlgo
from src.base import PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph
from src.utils import edge_ressources_snapshot, edge_capacity_ok, allocate_on_edges


P_STATIC = 200
P_CPU_UNIT = 5


class CSP(PlacementAlgo):
    """CP-SAT placement optimizing only energy (node energy)."""

    def place(self, service_graph: ServiceGraph, network_graph: NetworkGraph, **kwargs) -> PlacementResult:
        SG, NG = service_graph.G, network_graph.G

        comps = list(SG.nodes())
        hosts = list(NG.nodes())
        if not comps or not hosts:
            return PlacementResult(mapping={}, paths={}, meta={'status': 'failed', 'reason': 'empty_graph'})

        cpu_req = {c: int(SG.nodes[c].get('cpu') or 0) for c in comps}
        ram_req = {c: int(SG.nodes[c].get('ram') or 0) for c in comps}
        cpu_cap = {h: int(NG.nodes[h].get('cpu') or 0) for h in hosts}
        ram_cap = {h: int(NG.nodes[h].get('ram') or 0) for h in hosts}

        model = cp_model.CpModel()

        # Assignment variables: x[c][h] = 1 if component c placed on host h
        x: Dict[int, Dict[int, cp_model.IntVar]] = {}
        for c in comps:
            x[c] = {}
            for h in hosts:
                x[c][h] = model.NewBoolVar(f"x_{c}_{h}")
            model.Add(sum(x[c][h] for h in hosts) == 1)

        # Resource constraints per host
        cpu_used: Dict[int, cp_model.IntVar] = {}
        ram_used: Dict[int, cp_model.IntVar] = {}
        for h in hosts:
            cpu_used[h] = model.NewIntVar(0, cpu_cap[h], f"cpu_used_{h}")
            ram_used[h] = model.NewIntVar(0, ram_cap[h], f"ram_used_{h}")
            model.Add(cpu_used[h] == sum(cpu_req[c] * x[c][h] for c in comps))
            model.Add(ram_used[h] == sum(ram_req[c] * x[c][h] for c in comps))
            model.Add(cpu_used[h] <= cpu_cap[h])
            model.Add(ram_used[h] <= ram_cap[h])

        # Locality constraints if provided as pairs (component, host)
        dz = service_graph.metadata.get('component.DZ') or []
        if isinstance(dz, list) and len(dz) >= 2:
            if all(isinstance(v, int) for v in dz) and len(dz) % 2 == 0:
                for i in range(0, len(dz), 2):
                    c = dz[i]
                    h = dz[i + 1]
                    if c in comps and h in hosts:
                        model.Add(x[c][h] == 1)

        # Energy objective: node energy only
        energy_terms: List[cp_model.IntVar] = []
        for h in hosts:
            active = model.NewBoolVar(f"active_{h}")
            model.Add(cpu_used[h] > 0).OnlyEnforceIf(active)
            model.Add(cpu_used[h] == 0).OnlyEnforceIf(active.Not())

            static_e = model.NewIntVar(0, P_STATIC, f"static_e_{h}")
            model.Add(static_e == P_STATIC).OnlyEnforceIf(active)
            model.Add(static_e == 0).OnlyEnforceIf(active.Not())

            dyn_e = model.NewIntVar(0, cpu_cap[h] * P_CPU_UNIT, f"dyn_e_{h}")
            model.Add(dyn_e == cpu_used[h] * P_CPU_UNIT)

            energy_terms.extend([static_e, dyn_e])

        total_energy = model.NewIntVar(0, sum(cpu_cap.values()) * P_CPU_UNIT + len(hosts) * P_STATIC, "energy")
        model.Add(total_energy == sum(energy_terms))
        model.Minimize(total_energy)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = int(kwargs.get('num_workers', 8))
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return PlacementResult(mapping={}, paths={}, meta={'status': 'failed', 'reason': solver.StatusName(status)})

        mapping: Dict[int, int] = {}
        for c in comps:
            for h in hosts:
                if solver.Value(x[c][h]) == 1:
                    mapping[c] = h
                    break

        # Route edges with shortest path, check bandwidth constraints
        edge_res = edge_ressources_snapshot(network_graph)
        H = nx.DiGraph()
        for u, v, d in NG.edges(data=True):
            lat = float(d.get('latency') or 0.0)
            H.add_edge(u, v, weight=lat)

        paths: Dict[Tuple[int, int], List[int]] = {}
        for u, v, d in SG.edges(data=True):
            src_host = mapping[u]
            dst_host = mapping[v]
            bw_req = int(d.get('bandwidth') or 0)
            try:
                path = nx.shortest_path(H, source=src_host, target=dst_host, weight='weight')
            except nx.NetworkXNoPath:
                return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'no_path_{u}_{v}'})

            if not edge_capacity_ok(edge_res, path, bw_req):
                return PlacementResult(mapping=mapping, paths={}, meta={'status': 'failed', 'reason': f'constraints_{u}_{v}'})
            allocate_on_edges(edge_res, path, bw_req)
            paths[(u, v)] = path

        return PlacementResult(
            mapping=mapping,
            paths=paths,
            meta={
                'status': 'success',
                'energy': solver.Value(total_energy),
                'solver_status': solver.StatusName(status),
            },
        )
