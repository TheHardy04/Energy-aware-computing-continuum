from typing import Dict, Tuple, List, Any

import networkx as nx
from ortools.sat.python import cp_model

from src.placementAlgo import PlacementAlgo
from src.base import PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph
from src.utils import edge_ressources_snapshot, edge_capacity_ok, allocate_on_edges


P_STATIC = 200  # Static energy cost for activating a host
P_CPU_UNIT = 5  # Energy coefficient per CPU unit used
P_LINK_UNIT = 1  # Energy coefficient for bandwidth * latency^2


class CSP(PlacementAlgo):
    """
    CP-SAT placement optimizing energy (node energy + link energy).
    Base on the work of [Ait Salaht, Farah, and Nora Izri. "Optimisation Conjointe de la Réplication et du Placement des Microservices pour les Applications IoT dans le Fog Computing." ALGOTEL 2025-27èmes Rencontres Francophones sur les Aspects Algorithmiques des Télécommunications. 2025](https://hal.science/hal-05032320v1/document).
    - Decision variables:
        - x[c][h] = 1 if component c is placed on host h
        - flow[l][u][v] = 1 if service link l uses physical edge (u,v)
    - Constraints:
        - Each component placed on exactly one host
        - Host capacity constraints (CPU, RAM)
        - Locality constraints (DZ)
        - Flow conservation for each service link
        - Bandwidth capacity on physical edges
    - Objective: Minimize total energy = node energy + link energy
        - Node energy = sum over hosts of (P_STATIC * is_active + P_CPU_UNIT * cpu_used)
        - Link energy = sum over physical edges of (total_bandwidth_on_edge * latency^2)
    - Returns mapping and per-edge routing meta.
    """

    def place(self, service_graph: ServiceGraph, network_graph: NetworkGraph, **kwargs) -> PlacementResult:
        SG, NG = service_graph.G, network_graph.G

        comps = list(SG.nodes())
        hosts = list(NG.nodes())
        edges = list(SG.edges(data=True))
        
        if not comps or not hosts:
            return PlacementResult(mapping={}, paths={}, meta={'status': 'failed', 'reason': 'empty_graph'})

        cpu_req = {c: int(SG.nodes[c].get('cpu') or 0) for c in comps}
        ram_req = {c: int(SG.nodes[c].get('ram') or 0) for c in comps}
        cpu_cap = {h: int(NG.nodes[h].get('cpu') or 0) for h in hosts}
        ram_cap = {h: int(NG.nodes[h].get('ram') or 0) for h in hosts}

        model = cp_model.CpModel()

        # 1) Placement Variables
        # x[c][h] = 1 if component c placed on host h
        x: Dict[int, Dict[int, cp_model.IntVar]] = {}
        for c in comps:
            x[c] = {}
            for h in hosts:
                x[c][h] = model.NewBoolVar(f"x_{c}_{h}")
            model.Add(sum(x[c][h] for h in hosts) == 1)

        # 2) Resource Constraints (CPU, RAM)
        cpu_used: Dict[int, cp_model.IntVar] = {}
        ram_used: Dict[int, cp_model.IntVar] = {}
        for h in hosts:
            cpu_used[h] = model.NewIntVar(0, cpu_cap[h], f"cpu_used_{h}")
            ram_used[h] = model.NewIntVar(0, ram_cap[h], f"ram_used_{h}")
            
            # Link placement var to usage
            model.Add(cpu_used[h] == sum(cpu_req[c] * x[c][h] for c in comps))
            model.Add(ram_used[h] == sum(ram_req[c] * x[c][h] for c in comps))

        # 3) Locality Constraints
        dz = service_graph.metadata.get('component.DZ') or []
        if isinstance(dz, list) and len(dz) >= 2:
            if all(isinstance(v, int) for v in dz) and len(dz) % 2 == 0:
                for i in range(0, len(dz), 2):
                    c = dz[i]
                    h = dz[i + 1]
                    if c in comps and h in hosts:
                        model.Add(x[c][h] == 1)

        # 4) Routing / Flow Constraints
        # We model flow for each virtual link (service edge) separately to ensure connectivity.
        # This replaces the restrictive path pre-calculation.
        
        # Mapping for easy lookup
        topo_edges = list(NG.edges(data=True))
    
        # Max path length to prevent cycles/infinite loops (optional but good for solver)
        MAX_HOPS = len(hosts) 

        service_link_flows = {} 
        
        for l_idx, (sc, dc, d_s) in enumerate(edges):
            # For this service link, define flow variables on all network edges
            # flow[u][v] = 1 if traffic for (sc->dc) goes through physical link u->v
            
            flows = {}
            for u_n, v_n, _ in topo_edges:
                flows[(u_n, v_n)] = model.NewBoolVar(f"flow_{l_idx}_{u_n}_{v_n}")
            
            service_link_flows[l_idx] = flows

            # Flow conservation constraints for each host node
            for h in hosts:
                # Inflow: sum of flows entering h
                inflow = []
                for u_n, v_n, _ in topo_edges:
                    if v_n == h:
                        inflow.append(flows[(u_n, v_n)])
                
                # Outflow: sum of flows leaving h
                outflow = []
                for u_n, v_n, _ in topo_edges:
                    if u_n == h:
                        outflow.append(flows[(u_n, v_n)])
                
                # Net flow = Out - In
                # If h hosts Source (sc): Net = 1 (unless sc and dc on h)
                # If h hosts Dest (dc): Net = -1 (unless sc and dc on h)
                # Else: Net = 0
                
                is_src = x[sc][h]
                is_dst = x[dc][h]
                
                # If src and dst are on same host h, then is_src=1, is_dst=1 => Net=0. Correct.
                # If src on h, dst elsewhere => is_src=1, is_dst=0 => Net=1 (Out - In = 1). Correct.
                # If dst on h, src elsewhere => is_src=0, is_dst=1 => Net=-1 (Out - In = -1). Correct.
                
                model.Add(sum(outflow) - sum(inflow) == is_src - is_dst)

                # Loop prevention (simple version: sum of flows <= N-1?)
                # Or just rely on minimization of energy (paths will be short)
                # But we must ensure that if src == dst (same host), flow is 0 everywhere on external links?
                
                # Handling self-loop / same node placement:
                # If sc and dc are on the same host h, then is_src - is_dst = 0 at h, and 0 everywhere else.
                # The trivial solution flow=0 everywhere is valid.
                # However, is_src and is_dst don't force flow to be 0 if they cancel out.
                # But minimizing energy (link energy) will force flow to 0 if not needed.
                # So this should be fine.


        # 5) Bandwidth Capacity Constraints
        # Total flow on edge (u_n, v_n) must not exceed capacity
        for u_n, v_n, d_n in topo_edges:
            bw_val = d_n.get('bandwidth')
            # If bandwidth is -1 or None, treat as infinite (no constraint)
            if bw_val is None:
                continue
            capacity = int(bw_val)
            if capacity == -1:
                continue

            # Sum of bandwidths of all service links using this physical edge
            current_edge_load = []
            for l_idx, (sc, dc, d_s) in enumerate(edges):
                bw_req = int(d_s.get('bandwidth') or 0)
                # flow var specific to this edge
                f = service_link_flows[l_idx].get((u_n, v_n))
                if f is not None:
                    current_edge_load.append(f * bw_req)
            
            if current_edge_load:
                model.Add(sum(current_edge_load) <= capacity)

        # 6) Objective Function: Energy
        
        # A. Node Energy
        # E_node = sum(P_STATIC * is_active + E_dynamic * cpu_used)
        node_energy_vars = []
        for h in hosts:
            # is_active = 1 if cpu_used > 0
            is_active = model.NewBoolVar(f"active_{h}")
            model.Add(cpu_used[h] > 0).OnlyEnforceIf(is_active)
            model.Add(cpu_used[h] == 0).OnlyEnforceIf(is_active.Not())
            
            static_part = model.NewIntVar(0, P_STATIC, f"static_{h}")
            model.Add(static_part == P_STATIC).OnlyEnforceIf(is_active)
            model.Add(static_part == 0).OnlyEnforceIf(is_active.Not())
            
            dynamic_part = model.NewIntVar(0, cpu_cap[h] * P_CPU_UNIT, f"dyn_{h}")
            model.Add(dynamic_part == cpu_used[h] * P_CPU_UNIT)
            
            node_energy_vars.append(static_part)
            node_energy_vars.append(dynamic_part)
            
        total_node_energy = model.NewIntVar(0, sum(cpu_cap.values()) * P_CPU_UNIT + len(hosts) * P_STATIC, "total_node_energy")
        model.Add(total_node_energy == sum(node_energy_vars))
        
        # B. Link Energy
        # E_link = sum( flow_l * latency_e^2 ) for all service links l, network edges e
        # where flow_l is bandwidth of service link l
        
        # In the reference code:
        # total_link_energy = sum( flow_on_arc * lat_arc^2 )
        # where flow_on_arc is total bandwidth on that arc.
        # This is mathematically equivalent to sum( is_l_on_e * bw_l * lat_e^2 )
        
        link_energy_vars = []
        
        # Precompute max possible link energy to set bounds
        max_lat = max([float(d.get('latency') or 0) for _,_,d in topo_edges] + [1])
        # A rough upper bound
        max_link_E = int(len(edges) * max([d.get('bandwidth') or 0 for _,_,d in edges] + [1]) * (max_lat**2) * len(hosts))

        total_link_energy = model.NewIntVar(0, max_link_E * len(topo_edges), "total_link_energy")

        for u_n, v_n, d_n in topo_edges:
            lat = float(d_n.get('latency') or 0)
            if lat <= 0:
                continue
                
            lat_sq = int(lat * lat)
            
            # Flow on this specific physical edge
            edge_flow_vars = []
            for l_idx, (sc, dc, d_s) in enumerate(edges):
                bw_req = int(d_s.get('bandwidth') or 0)
                f = service_link_flows[l_idx].get((u_n, v_n))
                if f is not None:
                    edge_flow_vars.append(f * bw_req)
            
            if edge_flow_vars:
                # E_edge = TotalFlow * lat^2
                #        = Sum(f_l * bw_l) * lat^2
                #        = Sum(f_l * (bw_l * lat^2))
                
                # To keep it integer, ensure bw * lat^2 is int. 
                # bandwidth is int. lat might be float. 
                # Reference uses "lat * lat" (integer context likely or rounded).
                # We cast lat to int earlier if needed, or assume inputs are compatible.
                # Here we use lat_sq computed above.
                
                contrib = model.NewIntVar(0, max_link_E, f"link_E_{u_n}_{v_n}")
                model.Add(contrib == sum(edge_flow_vars) * lat_sq)
                link_energy_vars.append(contrib)

        if link_energy_vars:
            model.Add(total_link_energy == sum(link_energy_vars))
        else:
            model.Add(total_link_energy == 0)

        # Total Energy
        # Calculate a safe upper bound based on physical limits
        max_node_energy = sum(cpu_cap.values()) * P_CPU_UNIT + len(hosts) * P_STATIC
        total_energy_max = max_node_energy + (max_link_E * len(topo_edges))
        
        # Ensure it fits in CP-SAT limits (approx 2^62)
        # Using a safer upper bound to avoid overflow issues
        safe_max = (2**50) 
        total_energy_max = min(total_energy_max, safe_max)

        total_energy = model.NewIntVar(0, total_energy_max, "total_energy")
        model.Add(total_energy == total_node_energy + total_link_energy)

        # Minimize
        model.Minimize(total_energy)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = int(kwargs.get('num_workers', 8))
        # solver.parameters.log_search_progress = True
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if status == cp_model.MODEL_INVALID:
                print(f"Model Invalid: {model.Validate()}")
            return PlacementResult(
                mapping={}, 
                paths={}, 
                meta={'status': 'failed', 'reason': solver.StatusName(status)}
            )

        # Extract Results
        mapping: Dict[int, int] = {}
        for c in comps:
            for h in hosts:
                if solver.Value(x[c][h]) == 1:
                    mapping[c] = h
                    break

        paths: Dict[Tuple[int, int], List[int]] = {}
        for l_idx, (sc, dc, d_s) in enumerate(edges):
            # Reconstruct path from flow vars
            # Start at mapping[sc]
            curr = mapping[sc]
            target = mapping[dc]
            path = [curr]
            
            # Simple path reconstruction (greedy follow flow)
            # Warning: Flow loops or splits shouldn't happen in optimal energy sol (min hops usually), 
            # but we need to supply a list of nodes.
            
            if curr == target:
                paths[(sc, dc)] = [curr]
                continue

            visited = {curr}
            while curr != target:
                found_next = False
                for u_n, v_n, _ in topo_edges:
                    if u_n == curr:
                        val = solver.Value(service_link_flows[l_idx][(u_n, v_n)])
                        if val == 1:
                            if v_n in visited:
                                continue # Avoid cycles if any
                            curr = v_n
                            path.append(curr)
                            visited.add(curr)
                            found_next = True
                            break
                if not found_next:
                    # Could happen if solver output is weird or multipath?
                    # Fallback
                    break
            
            paths[(sc, dc)] = path

        return PlacementResult(
            mapping=mapping,
            paths=paths,
            meta={
                'status': 'success',
                'total_energy': solver.Value(total_energy),
                'node_energy': solver.Value(total_node_energy),
                'link_energy': solver.Value(total_link_energy),
                'solver_status': solver.StatusName(status),
            },
        )