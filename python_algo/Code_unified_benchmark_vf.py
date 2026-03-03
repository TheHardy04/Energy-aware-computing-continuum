#!/usr/bin/env python3
"""
Code Placement - Benchmark Unifie : CSP / LLM (OPRO) / Glouton
========================================================================
"""

import re, json, time, random, heapq, os, sys, argparse
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field

# =====================================================================
# CONSTANTES
# =====================================================================
P_STATIC = 200
P_CPU_UNIT = 5
NORM = 10000
INF_V = 10 ** 9
P_CPU_LOCAL = float(os.environ.get("P_CPU_WATTS", "45"))  # Watts

# Energie par 1k tokens (Wh) - source: Luccioni et al. 2023
ENERGY_PER_1K_TOKENS = {
    "anthropic": 0.004,
    "openai": 0.004,
    "gemini": 0.003,
    "ollama": 0.0,  # mesure P*T
}
# Cout USD par 1k tokens (input, output)
COST_PER_1K_TOKENS = {
    "anthropic": (0.003, 0.015),
    "openai": (0.005, 0.015),
    "gemini": (0.0005, 0.0015),
    "ollama": (0.0, 0.0),
}


# =====================================================================
# PARSERS (communs aux deux codes)
# =====================================================================
def parse_properties(path):
    props = {}
    with open(path, encoding="utf-8") as f:
        cont = ""
        for raw in f:
            line = raw.rstrip("\n")
            if line.endswith("\\"): cont += line[:-1]; continue
            line = (cont + line).strip(); cont = ""
            if not line or line[0] in "#!": continue
            if "=" in line: k, v = line.split("=", 1)
            elif ":" in line: k, v = line.split(":", 1)
            else: continue
            props[k.strip()] = v.strip()
    return props


def extract_ints(s):
    return list(map(int, re.findall(r"-?\d+", s)))


# =====================================================================
# DATA STRUCTURES
# =====================================================================
@dataclass
class InfraData:
    nbH: int = 0
    diameter: int = 0
    cpuCap: List[int] = field(default_factory=list)
    ramCap: List[int] = field(default_factory=list)
    hostCost: List[int] = field(default_factory=list)
    adj: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)
    arc_bw: Dict[Tuple[int, int], int] = field(default_factory=dict)
    arc_lat: Dict[Tuple[int, int], int] = field(default_factory=dict)
    sp_latency: List[List[int]] = field(default_factory=list)
    sp_path: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)


@dataclass
class AppliData:
    nbS: int = 0; nbC: int = 0
    cpuComp: List[int] = field(default_factory=list)
    ramComp: List[int] = field(default_factory=list)
    dataFlowRates: List[int] = field(default_factory=list)
    muRates: List[int] = field(default_factory=list)
    nbL: int = 0
    linkPerServ: List[Tuple[int, int]] = field(default_factory=list)
    bdwPair: List[int] = field(default_factory=list)
    latPair: List[int] = field(default_factory=list)
    nbDZ: int = 0
    DZ: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class NormBounds:
    lat_ub: int = 1
    cost_max: int = 1
    energy_ub: int = 1
    e_link_ub: int = 1
    e_node_ub: int = 1


@dataclass
class PlacementSolution:
    placement: Dict[int, int] = field(default_factory=dict)
    paths: Dict[int, List[int]] = field(default_factory=dict)
    feasible: bool = False
    violations: List[str] = field(default_factory=list)
    worst_case_latency: int = 0
    total_latency: int = 0
    infra_cost: int = 0
    energy_link: int = 0
    energy_node: int = 0
    energy_total: int = 0
    active_hosts: Set[int] = field(default_factory=set)
    host_cpu_used: Dict[int, int] = field(default_factory=dict)
    host_ram_used: Dict[int, int] = field(default_factory=dict)
    total_flow_on_arc: Dict[Tuple[int, int], int] = field(default_factory=dict)
    consumed_lat: Dict[int, int] = field(default_factory=dict)
    objective: float = float("inf")
    objective_int: int = 0
    solve_time: float = 0.0


@dataclass
class SolverResult:
    """Resultat unifie pour le tableau comparatif."""
    name: str = ""
    sol: PlacementSolution = field(default_factory=PlacementSolution)
    # Cout du solveur
    t_total_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    e_solver_wh: float = 0.0
    cost_usd: float = 0.0
    e_method: str = ""  # "P*T", "tokens*eps", "~0"


# =====================================================================
# LOADERS
# =====================================================================
def load_infra(path):
    p = parse_properties(path)
    d = InfraData()
    d.nbH = int(p["hosts.nb"])
    d.diameter = int(p.get("network.diameter", d.nbH - 1))
    cfg = extract_ints(p["hosts.configuration"])
    d.cpuCap = [cfg[2 * i] for i in range(d.nbH)]
    d.ramCap = [cfg[2 * i + 1] for i in range(d.nbH)]
    if "hosts.cost" in p:
        v = extract_ints(p["hosts.cost"])
        d.hostCost = v[:d.nbH]
        while len(d.hostCost) < d.nbH:
            d.hostCost.append(d.cpuCap[len(d.hostCost)])
    else:
        d.hostCost = list(d.cpuCap)
    topo = extract_ints(p["network.topology"])
    d.adj = defaultdict(list)
    for i in range(0, len(topo), 4):
        u, v, bw, lat = topo[i:i + 4]
        if u != v:
            d.adj[u].append((v, bw, lat))
            d.arc_bw[(u, v)] = bw
            d.arc_lat[(u, v)] = lat
    # Dijkstra all-pairs
    d.sp_latency = [[INF_V] * d.nbH for _ in range(d.nbH)]
    for src in range(d.nbH):
        d.sp_latency[src][src] = 0
        dist = [INF_V] * d.nbH; dist[src] = 0; prev = [-1] * d.nbH
        pq = [(0, src)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist[u]: continue
            for v, bw, lat in d.adj[u]:
                nd = dd + lat
                if nd < dist[v]:
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        d.sp_latency[src] = dist
        for dst in range(d.nbH):
            if dist[dst] < INF_V and dst != src:
                pa = []; n = dst
                while n != -1: pa.append(n); n = prev[n]
                pa.reverse(); d.sp_path[(src, dst)] = pa
    return d


def load_appli(path):
    p = parse_properties(path)
    d = AppliData()
    comps = extract_ints(p["application.components"])
    d.nbS = int(p["application.nb"]); d.nbC = sum(comps)
    reqs = extract_ints(p["components.requirements"])
    d.cpuComp = [reqs[4 * i] for i in range(d.nbC)]
    d.ramComp = [reqs[4 * i + 1] for i in range(d.nbC)]
    d.dataFlowRates = [reqs[4 * i + 2] for i in range(d.nbC)]
    d.muRates = [reqs[4 * i + 3] for i in range(d.nbC)]
    links = extract_ints(p["links.description"])
    d.nbL = int(p["links.nb"])
    for i in range(d.nbL):
        _, s, t, bw, lat = links[5 * i:5 * i + 5]
        d.linkPerServ.append((s, t)); d.bdwPair.append(bw); d.latPair.append(lat)
    d.nbDZ = int(p["component.nbDZ"])
    if d.nbDZ > 0:
        dz = extract_ints(p.get("component.DZ", ""))
        for i in range(d.nbDZ):
            d.DZ.append((dz[2 * i], dz[2 * i + 1]))
    return d


def compute_norm_bounds(infra, appli):
    b = NormBounds()
    max_arc_lat = max(infra.arc_lat.values(), default=1)
    b.lat_ub = max(1, appli.nbL * infra.diameter * max_arc_lat)
    b.cost_max = max(1, sum(infra.hostCost))
    max_flow = max(1, sum(appli.bdwPair))
    b.e_link_ub = max(1, max_flow * max_arc_lat * max_arc_lat)
    b.e_node_ub = infra.nbH * (P_STATIC + max(infra.cpuCap) * P_CPU_UNIT)
    b.energy_ub = b.e_link_ub + b.e_node_ub
    return b


# =====================================================================
# EVALUATEUR UNIFIE (utilise par toutes les approches sauf CSP)
# =====================================================================
def evaluate_solution(placement, infra, appli, bounds,
                      w_lat=0, w_cost=0, w_energy=1):
    sol = PlacementSolution(); sol.placement = dict(placement)
    sol.violations = []

    # C0: Completude
    for c in range(appli.nbC):
        if c not in placement:
            sol.violations.append(f"C0: Composant {c} non place")
            sol.placement[c] = 0

    # C1: Localite DZ
    for ci, hi in appli.DZ:
        if sol.placement.get(ci) != hi:
            sol.violations.append(f"C1: DZ C{ci} devrait etre sur H{hi}, est sur H{sol.placement.get(ci)}")

    # C2: Capacite
    cu = [0] * infra.nbH; ru = [0] * infra.nbH
    for c in range(appli.nbC):
        h = sol.placement[c]
        cu[h] += appli.cpuComp[c]; ru[h] += appli.ramComp[c]
    for h in range(infra.nbH):
        if cu[h] > infra.cpuCap[h]:
            sol.violations.append(f"C2: CPU H{h}: {cu[h]}/{infra.cpuCap[h]}")
        if ru[h] > infra.ramCap[h]:
            sol.violations.append(f"C2: RAM H{h}: {ru[h]}/{infra.ramCap[h]}")
    sol.host_cpu_used = {h: cu[h] for h in range(infra.nbH)}
    sol.host_ram_used = {h: ru[h] for h in range(infra.nbH)}
    sol.active_hosts = {h for h in range(infra.nbH) if cu[h] > 0}

    # C3: Chemins + latence par lien
    link_lats = []
    arc_flow = defaultdict(int)
    for l in range(appli.nbL):
        sC, tC = appli.linkPerServ[l]
        hs, ht = sol.placement[sC], sol.placement[tC]
        if hs == ht:
            sol.paths[l] = [hs]; sol.consumed_lat[l] = 0; link_lats.append(0)
        else:
            path = infra.sp_path.get((hs, ht), [])
            if not path:
                sol.violations.append(f"C3: Pas de chemin L{l}: H{hs}->H{ht}")
                sol.consumed_lat[l] = INF_V; link_lats.append(INF_V)
            else:
                sol.paths[l] = path
                sol.consumed_lat[l] = infra.sp_latency[hs][ht]
                link_lats.append(infra.sp_latency[hs][ht])
                for i in range(len(path) - 1):
                    arc_flow[(path[i], path[i + 1])] += appli.bdwPair[l]
    sol.total_flow_on_arc = dict(arc_flow)

    # C4: Bande passante
    for (u, v), flow in arc_flow.items():
        bw_cap = infra.arc_bw.get((u, v), 0)
        if bw_cap > 0 and flow > bw_cap:
            sol.violations.append(f"C4: BW ({u},{v}): {flow}>{bw_cap}")

    # Metriques
    sol.worst_case_latency = max(link_lats) if link_lats else 0
    sol.total_latency = sum(x for x in link_lats if x < INF_V)
    sol.infra_cost = sum(infra.hostCost[h] for h in sol.active_hosts)
    sol.energy_link = sum(
        fl * infra.arc_lat.get((u, v), 0) ** 2
        for (u, v), fl in arc_flow.items())
    sol.energy_node = sum(P_STATIC + cu[h] * P_CPU_UNIT for h in sol.active_hosts)
    sol.energy_total = sol.energy_link + sol.energy_node
    sol.feasible = len(sol.violations) == 0

    # Objectif normalise (SOMME bout-en-bout dans le terme latence)
    if sol.feasible:
        def norm_int(val, ub):
            return (val * NORM) // ub if ub > 0 else 0
        lat_n = norm_int(sol.total_latency, bounds.lat_ub)
        cost_n = norm_int(sol.infra_cost, bounds.cost_max)
        ener_n = norm_int(sol.energy_total, bounds.energy_ub)
        W = 1000
        wl, wc, we = int(round(w_lat * W)), int(round(w_cost * W)), int(round(w_energy * W))
        sol.objective_int = lat_n * wl + cost_n * wc + ener_n * we
        sol.objective = float(sol.objective_int)
    else:
        sol.objective_int = 10 ** 9; sol.objective = float("inf")
    return sol


# =====================================================================
# GLOUTON (FFD) - baseline
# =====================================================================
def greedy_placement(infra, appli):
    pl = {}; cu = [0] * infra.nbH; ru = [0] * infra.nbH
    for ci, hi in appli.DZ:
        pl[ci] = hi; cu[hi] += appli.cpuComp[ci]; ru[hi] += appli.ramComp[ci]
    for c in range(appli.nbC):
        if c in pl: continue
        bh = -1; bs = float("inf")
        for h in range(infra.nbH):
            if cu[h] + appli.cpuComp[c] > infra.cpuCap[h]: continue
            if ru[h] + appli.ramComp[c] > infra.ramCap[h]: continue
            cost_pen = infra.hostCost[h] if cu[h] == 0 else 0
            lat_pen = 0
            for nb in pl:
                for l in range(appli.nbL):
                    sC, tC = appli.linkPerServ[l]
                    if (sC == c and tC == nb) or (tC == c and sC == nb):
                        lat_pen += infra.sp_latency[h][pl[nb]] if infra.sp_latency[h][pl[nb]] < INF_V else 9999
            sc = cost_pen * 10 + lat_pen
            if sc < bs: bs = sc; bh = h
        pl[c] = bh if bh >= 0 else 0
        if bh >= 0:
            cu[bh] += appli.cpuComp[c]; ru[bh] += appli.ramComp[c]
    return pl


def random_placement(infra, appli):
    pl = {}; cu = [0] * infra.nbH; ru = [0] * infra.nbH
    for ci, hi in appli.DZ:
        pl[ci] = hi; cu[hi] += appli.cpuComp[ci]; ru[hi] += appli.ramComp[ci]
    hosts = list(range(infra.nbH))
    for c in range(appli.nbC):
        if c in pl: continue
        random.shuffle(hosts)
        placed = False
        for h in hosts:
            if (cu[h] + appli.cpuComp[c] <= infra.cpuCap[h] and
                    ru[h] + appli.ramComp[c] <= infra.ramCap[h]):
                pl[c] = h; cu[h] += appli.cpuComp[c]; ru[h] += appli.ramComp[c]
                placed = True; break
        if not placed:
            pl[c] = 0
    return pl


# =====================================================================
# REPARATION
# =====================================================================
def repair_placement(placement, infra, appli):
    pl = dict(placement)
    locked = {ci for ci, _ in appli.DZ}
    # C1: forcer DZ
    for ci, hi in appli.DZ:
        pl[ci] = hi
    # C0: completer
    for c in range(appli.nbC):
        if c not in pl: pl[c] = 0
    # C2: deplacer les composants en surplus
    for _ in range(appli.nbC * 2):
        cu = [0] * infra.nbH; ru = [0] * infra.nbH
        for c in range(appli.nbC):
            cu[pl[c]] += appli.cpuComp[c]; ru[pl[c]] += appli.ramComp[c]
        worst_h = -1; worst_ov = 0
        for h in range(infra.nbH):
            ov = max(0, cu[h] - infra.cpuCap[h]) + max(0, ru[h] - infra.ramCap[h])
            if ov > worst_ov: worst_ov = ov; worst_h = h
        if worst_h == -1: break
        movable = sorted(
            [c for c in range(appli.nbC) if pl[c] == worst_h and c not in locked],
            key=lambda c: -(appli.cpuComp[c] + appli.ramComp[c]))
        if not movable: break
        moved = False
        for c in movable:
            for h2 in range(infra.nbH):
                if h2 == worst_h: continue
                if cu[h2] + appli.cpuComp[c] > infra.cpuCap[h2]: continue
                if ru[h2] + appli.ramComp[c] > infra.ramCap[h2]: continue
                pl[c] = h2; moved = True; break
            if moved: break
        if not moved: break
    # C3: connectivite
    for l in range(appli.nbL):
        sC, tC = appli.linkPerServ[l]
        hs, ht = pl[sC], pl[tC]
        if hs != ht and not infra.sp_path.get((hs, ht)):
            if tC not in locked: pl[tC] = hs
            elif sC not in locked: pl[sC] = ht
    return pl


# =====================================================================
# DAG ANALYSIS HELPERS
# =====================================================================
def _build_dag_adjacency(appli):
    fwd = defaultdict(list); rev = defaultdict(list)
    for l, (s, t) in enumerate(appli.linkPerServ):
        fwd[s].append((t, l, appli.bdwPair[l]))
        rev[t].append((s, l, appli.bdwPair[l]))
    return fwd, rev


def _find_tail_components(appli, dz_map, fwd, rev, max_depth=2):
    """A7: Find tail components (pure chain to sink DZ, max 2 hops)."""
    dz_set = set(dz_map.keys())
    tails = {}
    for ci, hi in dz_map.items():
        if any(t not in dz_set for t, _, _ in fwd.get(ci, [])):
            continue  # not a sink DZ
        visited = {ci}
        queue = deque([(ci, 0)])
        while queue:
            c, depth = queue.popleft()
            if depth >= max_depth: continue
            for pred, _, _ in rev.get(c, []):
                if pred in visited or pred in dz_set: continue
                if all(t in visited for t, _, _ in fwd.get(pred, [])):
                    visited.add(pred); queue.append((pred, depth + 1))
        tail_comps = visited - dz_set
        if tail_comps: tails[ci] = (hi, tail_comps)
    return tails



# =====================================================================
# STANDALONE LOCAL SEARCH (for LLM post-processing)
# =====================================================================
def standalone_local_search(start_pl, infra, appli, bounds, w, max_iter=150):
    """Local search : Moves + swaps + pair-relocate + LNS + exhaustive ruin, with capacity pre-checks.
    """
    import math

    locked = {ci for ci, _ in appli.DZ}
    dz_map = {ci: hi for ci, hi in appli.DZ}
    dz_hosts = [hi for _, hi in appli.DZ]
    comp_graph = defaultdict(list)
    for l in range(appli.nbL):
        s, t = appli.linkPerServ[l]; bw = appli.bdwPair[l]
        comp_graph[s].append((t, bw)); comp_graph[t].append((s, bw))

    # Pre-compute hosts nearest to DZ hosts
    near_dz_hosts = set()
    for dh in dz_hosts:
        ranked = sorted(range(infra.nbH),
                        key=lambda h: infra.sp_latency[dh][h] if infra.sp_latency[dh][h] < INF_V else 999999)
        k = min(6, max(3, infra.nbH // 2))
        for h in ranked[:k]:
            near_dz_hosts.add(h)

    eval_cache = {}

    def _ev(pl):
        key = tuple(sorted(pl.items()))
        if key in eval_cache:
            return eval_cache[key]
        result = evaluate_solution(pl, infra, appli, bounds, *w)
        eval_cache[key] = result
        return result

    def _fits(h, c, cu, ru):
        return (cu[h] + appli.cpuComp[c] <= infra.cpuCap[h] and
                ru[h] + appli.ramComp[c] <= infra.ramCap[h])

    best_pl = dict(start_pl); best = _ev(best_pl)
    curr_pl = dict(best_pl); curr = best
    t0 = time.time(); no_imp = 0
    movable = [c for c in range(appli.nbC) if c not in locked]

    # Maintain capacity arrays incrementally
    cu = [0] * infra.nbH; ru = [0] * infra.nbH
    for c in range(appli.nbC):
        h = curr_pl[c]
        cu[h] += appli.cpuComp[c]; ru[h] += appli.ramComp[c]

    def _candidate_hosts(comp):
        cands = set()
        for h in range(infra.nbH):
            if cu[h] > 0: cands.add(h)
        for dh in dz_hosts: cands.add(dh)
        for nb, _ in comp_graph[comp]:
            cands.add(curr_pl[nb])
        cands.update(near_dz_hosts)
        return cands

    sa_temp = max(1.0, best.objective * 0.01) if best.feasible else 1000.0
    sa_cool = 0.92

    for it in range(max_iter):
        improved = False

        # Phase 1: Single moves (first-improvement + capacity pre-check)
        random.shuffle(movable)
        for c in movable:
            if improved: break
            old_h = curr_pl[c]
            cpu_c = appli.cpuComp[c]; ram_c = appli.ramComp[c]
            candidates = _candidate_hosts(c)
            for h in candidates:
                if h == old_h: continue
                if cu[h] + cpu_c > infra.cpuCap[h]: continue
                if ru[h] + ram_c > infra.ramCap[h]: continue
                cand = dict(curr_pl); cand[c] = h; s = _ev(cand)
                if s.feasible and s.objective < curr.objective:
                    cu[old_h] -= cpu_c; ru[old_h] -= ram_c
                    cu[h] += cpu_c; ru[h] += ram_c
                    curr_pl = cand; curr = s
                    if curr.objective < best.objective:
                        best_pl = dict(curr_pl); best = curr
                    improved = True; break

        # Phase 2: Swaps (first-improvement + capacity pre-check)
        if not improved:
            swap_found = False
            for i in range(len(movable)):
                if swap_found: break
                for j in range(i + 1, len(movable)):
                    ci, cj = movable[i], movable[j]
                    hi, hj = curr_pl[ci], curr_pl[cj]
                    if hi == hj: continue
                    cpu_i, cpu_j = appli.cpuComp[ci], appli.cpuComp[cj]
                    ram_i, ram_j = appli.ramComp[ci], appli.ramComp[cj]
                    if cu[hi] - cpu_i + cpu_j > infra.cpuCap[hi]: continue
                    if ru[hi] - ram_i + ram_j > infra.ramCap[hi]: continue
                    if cu[hj] - cpu_j + cpu_i > infra.cpuCap[hj]: continue
                    if ru[hj] - ram_j + ram_i > infra.ramCap[hj]: continue
                    cand = dict(curr_pl)
                    cand[ci], cand[cj] = hj, hi
                    s = _ev(cand)
                    if s.feasible and s.objective < curr.objective:
                        cu[hi] += (cpu_j - cpu_i); ru[hi] += (ram_j - ram_i)
                        cu[hj] += (cpu_i - cpu_j); ru[hj] += (ram_i - ram_j)
                        curr_pl = cand; curr = s
                        if curr.objective < best.objective:
                            best_pl = dict(curr_pl); best = curr
                        improved = True; swap_found = True; break

        # Phase 3: Pair-relocate
        if not improved and len(movable) <= 15:
            for ci in movable:
                for cj in movable:
                    if ci >= cj: continue
                    linked = any(nb == cj for nb, _ in comp_graph[ci])
                    if not linked: continue
                    cpu_pair = appli.cpuComp[ci] + appli.cpuComp[cj]
                    ram_pair = appli.ramComp[ci] + appli.ramComp[cj]
                    for h in range(infra.nbH):
                        if h == curr_pl[ci] and h == curr_pl[cj]: continue
                        used_cpu = cu[h]; used_ram = ru[h]
                        if curr_pl[ci] == h: used_cpu -= appli.cpuComp[ci]; used_ram -= appli.ramComp[ci]
                        if curr_pl[cj] == h: used_cpu -= appli.cpuComp[cj]; used_ram -= appli.ramComp[cj]
                        if used_cpu + cpu_pair > infra.cpuCap[h]: continue
                        if used_ram + ram_pair > infra.ramCap[h]: continue
                        cand = dict(curr_pl); cand[ci] = h; cand[cj] = h
                        s = _ev(cand)
                        if s.feasible and s.objective < best.objective:
                            old_hi, old_hj = curr_pl[ci], curr_pl[cj]
                            cu[old_hi] -= appli.cpuComp[ci]; ru[old_hi] -= appli.ramComp[ci]
                            cu[old_hj] -= appli.cpuComp[cj]; ru[old_hj] -= appli.ramComp[cj]
                            cu[h] += cpu_pair; ru[h] += ram_pair
                            curr_pl = cand; curr = s
                            best_pl = dict(cand); best = s; improved = True; break
                    if improved: break
                if improved: break

        # Phase 4: LNS with SA acceptance
        if not improved and len(movable) >= 3:
            base_k = min(3 + no_imp, min(7, len(movable)))
            lns_budget = min(60, len(movable) * 6)
            for lns_trial in range(lns_budget):
                k = random.randint(2, base_k)
                destroyed = random.sample(movable, k)
                cand = dict(curr_pl)
                lns_cu = [0] * infra.nbH; lns_ru = [0] * infra.nbH
                for c2 in range(appli.nbC):
                    if c2 not in destroyed:
                        lns_cu[cand[c2]] += appli.cpuComp[c2]
                        lns_ru[cand[c2]] += appli.ramComp[c2]
                order = sorted(destroyed, key=lambda c: -sum(
                    bw for _, bw in comp_graph[c]))
                ok = True
                for c in order:
                    bh = -1; bs = float("inf")
                    hosts = list(range(infra.nbH))
                    if lns_trial > lns_budget // 2:
                        random.shuffle(hosts)
                    for h in hosts:
                        if not _fits(h, c, lns_cu, lns_ru): continue
                        sc = 0
                        for nb, bw in comp_graph[c]:
                            nh = cand[nb]
                            d = infra.sp_latency[h][nh]
                            sc += bw * (d * d if d < INF_V else 999999)
                        if lns_cu[h] == 0 and h not in set(dz_map.values()):
                            sc += infra.hostCost[h] * 300
                        sc += random.uniform(0, sc * 0.15 * (1 + no_imp * 0.1))
                        if sc < bs: bs = sc; bh = h
                    if bh >= 0:
                        cand[c] = bh
                        lns_cu[bh] += appli.cpuComp[c]
                        lns_ru[bh] += appli.ramComp[c]
                    else:
                        ok = False; break
                if ok:
                    s = _ev(cand)
                    if s.feasible:
                        delta = s.objective - curr.objective
                        if delta < 0:
                            cu = [0] * infra.nbH; ru = [0] * infra.nbH
                            for c2 in range(appli.nbC):
                                cu[cand[c2]] += appli.cpuComp[c2]
                                ru[cand[c2]] += appli.ramComp[c2]
                            curr_pl = cand; curr = s
                            if s.objective < best.objective:
                                best_pl = dict(cand); best = s
                            improved = True; break
                        elif sa_temp > 0.1 and delta > 0:
                            prob = math.exp(-delta / max(sa_temp, 0.01))
                            if random.random() < prob:
                                cu = [0] * infra.nbH; ru = [0] * infra.nbH
                                for c2 in range(appli.nbC):
                                    cu[cand[c2]] += appli.cpuComp[c2]
                                    ru[cand[c2]] += appli.ramComp[c2]
                                curr_pl = cand; curr = s

        # Phase 5: Exhaustive ruin-2,3
        if not improved and len(movable) <= 15:
            from itertools import combinations as _combs
            active_hosts = sorted(set(best_pl.values()) | set(dz_map.values()))
            for h in sorted(range(infra.nbH), key=lambda h: infra.hostCost[h]):
                if h not in active_hosts: active_hosts.append(h)
                if len(active_hosts) >= 10: break
            nH = len(active_hosts)
            for ruin_size in [2, 3]:
                if improved: break
                if nH ** ruin_size > 25000: continue
                for ruined in _combs(movable, ruin_size):
                    if improved: break
                    ruin_set = set(ruined)
                    base_cu = [0] * infra.nbH; base_ru = [0] * infra.nbH
                    for c2 in range(appli.nbC):
                        if c2 not in ruin_set:
                            base_cu[best_pl[c2]] += appli.cpuComp[c2]
                            base_ru[best_pl[c2]] += appli.ramComp[c2]
                    for code in range(nH ** ruin_size):
                        hosts = []; c = code
                        for _ in range(ruin_size):
                            hosts.append(active_hosts[c % nH]); c //= nH
                        if all(hosts[i] == best_pl[ruined[i]] for i in range(ruin_size)): continue
                        cap_ok = True
                        test_cu = list(base_cu); test_ru = list(base_ru)
                        for idx in range(ruin_size):
                            h = hosts[idx]; comp = ruined[idx]
                            test_cu[h] += appli.cpuComp[comp]
                            test_ru[h] += appli.ramComp[comp]
                            if test_cu[h] > infra.cpuCap[h] or test_ru[h] > infra.ramCap[h]:
                                cap_ok = False; break
                        if not cap_ok: continue
                        cand = dict(best_pl)
                        for idx in range(ruin_size): cand[ruined[idx]] = hosts[idx]
                        s = _ev(cand)
                        if s.feasible and s.objective < best.objective:
                            best_pl = cand; best = s
                            curr_pl = dict(cand); curr = s
                            cu = [0] * infra.nbH; ru = [0] * infra.nbH
                            for c2 in range(appli.nbC):
                                cu[cand[c2]] += appli.cpuComp[c2]
                                ru[cand[c2]] += appli.ramComp[c2]
                            improved = True; break

        sa_temp *= sa_cool
        if improved: no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 3: break

    best.solve_time = time.time() - t0
    return best

# =====================================================================
# CSP SOLVER (CP-SAT)
# =====================================================================
def run_csp(infra_file, appli_file, w, bounds, infra, appli, verbose=True):
    """Lance le CSP et retourne un SolverResult unifie."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        print("  [SKIP] OR-Tools non installe")
        return None

    if verbose: print("\n  Construction du modele CP-SAT...")

    # On importe et utilise la classe CSP existante
    # Re-implementation legere inline pour eviter la dependance fichier
    csp_file = os.path.join(os.path.dirname(__file__), "Code_placement_csp_vf.py")
    if os.path.exists(csp_file):
        import importlib.util
        spec = importlib.util.spec_from_file_location("csp_mod", csp_file)
        csp_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(csp_mod)

        t0 = time.time()
        sol = csp_mod.TPDS2_Placement()
        sol.load_data(infra_file, appli_file)
        t_build_start = time.time()
        sol.declare_variables()
        sol.declare_constraints()
        sol.build_objective(w_lat=w[0], w_cost=w[1], w_energy=w[2])
        t_build = time.time() - t_build_start

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.max_time_in_seconds = 120

        if verbose: print(f"  Resolution... (timeout 120s)")
        t_solve_start = time.time()
        status = solver.Solve(sol.model)
        t_solve = time.time() - t_solve_start
        t_total = time.time() - t0

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"  [CSP] Pas de solution: {solver.StatusName(status)}")
            return None

        # Convertir en PlacementSolution unifie
        ps = PlacementSolution()
        ps.placement = {c: solver.Value(sol.h[c]) for c in range(sol.nbC)}
        ps.feasible = True
        ps.worst_case_latency = solver.Value(sol.worst_case_latency)
        ps.total_latency = solver.Value(sol.total_latency)
        ps.infra_cost = solver.Value(sol.total_deployment_cost)
        ps.energy_link = solver.Value(sol.total_link_energy)
        ps.energy_node = solver.Value(sol.total_node_energy)
        ps.energy_total = solver.Value(sol.total_energy_global)
        ps.objective = solver.ObjectiveValue()
        ps.objective_int = int(solver.ObjectiveValue())
        ps.solve_time = t_total
        ps.active_hosts = {h for h in range(sol.nbH) if solver.Value(sol.hcpu[h]) > 0}
        ps.host_cpu_used = {h: solver.Value(sol.hcpu[h]) for h in range(sol.nbH)}
        ps.host_ram_used = {h: solver.Value(sol.hram[h]) for h in range(sol.nbH)}

        # Chemins et latences par lien
        for l in range(sol.nbL):
            path_nodes = []
            for k in range(sol.diameter + 1):
                nd = solver.Value(sol.n[l][k])
                if nd == sol.FHOST: break
                path_nodes.append(nd)
            ps.paths[l] = path_nodes
            ps.consumed_lat[l] = solver.Value(sol.ConsumedLat[l])

        e_solver_wh = P_CPU_LOCAL * t_total / 3600.0
        stat = solver.StatusName(status)
        if verbose:
            print(f"  [CSP] {stat} | Obj={ps.objective:.0f} Lat_e2e={ps.total_latency}ms "
                  f"E={ps.energy_total} | t={t_total:.2f}s E_solver={e_solver_wh:.6f}Wh")

        return SolverResult(
            name=f"CSP ({stat})", sol=ps,
            t_total_s=t_total, e_solver_wh=e_solver_wh,
            e_method=f"P*T ({P_CPU_LOCAL:.0f}W)")
    else:
        print(f"  [SKIP] {csp_file} non trouve")
        return None


# =====================================================================
# LLM SOLVER
# =====================================================================
def _estimate_tokens(text):
    return max(1, len(text) // 4)


def _call_llm_with_tokens(prompt, system, provider, model, temperature=0.3):
    """Appelle le LLM et retourne (text, tokens_in, tokens_out).
    Supports: anthropic, openai, gemini (SDK + REST fallback), ollama."""
    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=4096, system=system,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        ti = r.usage.input_tokens if hasattr(r, 'usage') else _estimate_tokens(system + prompt)
        to = r.usage.output_tokens if hasattr(r, 'usage') else _estimate_tokens(r.content[0].text)
        return r.content[0].text, ti, to

    elif provider == "openai":
        import openai
        r = openai.OpenAI().chat.completions.create(
            model=model, max_tokens=4096, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}])
        ti = r.usage.prompt_tokens if r.usage else _estimate_tokens(system + prompt)
        to = r.usage.completion_tokens if r.usage else _estimate_tokens(r.choices[0].message.content)
        return r.choices[0].message.content, ti, to

    elif provider == "gemini":
        # Try SDK first, then REST fallback (v12-style, no dependency)
        try:
            from google import genai
            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", ""))
            r = client.models.generate_content(model=model, contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system, max_output_tokens=4096, temperature=temperature))
            ti = getattr(r.usage_metadata, 'prompt_token_count', None) or _estimate_tokens(system + prompt)
            to = getattr(r.usage_metadata, 'candidates_token_count', None) or _estimate_tokens(r.text)
            return r.text, ti, to
        except ImportError:
            # REST fallback (v12-style — no SDK needed)
            import urllib.request
            api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
            for api_version in ["v1beta", "v1alpha"]:
                url = (f"https://generativelanguage.googleapis.com/{api_version}/models/{model}"
                       f":generateContent?key={api_key}")
                payload = json.dumps({
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096}
                }).encode("utf-8")
                req = urllib.request.Request(url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode())
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                text = parts[0].get("text", "")
                                ti = _estimate_tokens(system + prompt)
                                to = _estimate_tokens(text)
                                return text, ti, to
                except urllib.error.HTTPError as e:
                    if e.code == 404 and api_version == "v1beta": continue
                    raise
            raise ValueError(f"Gemini API failed for model {model}")

    elif provider == "ollama":
        import urllib.request
        base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": temperature, "num_predict": 4096}}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        ti = data.get("prompt_eval_count", _estimate_tokens(system + prompt))
        to = data.get("eval_count", _estimate_tokens(data["message"]["content"]))
        return data["message"]["content"], ti, to

    raise ValueError(f"Provider inconnu: {provider}")


def _parse_llm_json(response, nbC):
    for pat in [r'```(?:json)?\s*(\{.*?\})\s*```',
                r'(\{[^{}]*"placement"\s*:\s*\{[^}]+\}[^}]*\})',
                r'(\{.*\})']:
        m = re.search(pat, response, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1) if '```' in pat else m.group(0))
                raw = data.get("placement", data)
                return {int(k): int(v) for k, v in raw.items()}
            except (json.JSONDecodeError, ValueError):
                continue
    return None


# =====================================================================
# A1: SYSTEM PROMPT ENRICHI (Chain-of-Thought structure)
# =====================================================================
LLM_SYSTEM = """You are an expert optimizer for the Service Placement Problem (SPP) in Fog Computing.
Your goal is to find a placement that MINIMIZES the objective value. Lower is better.

OBJECTIVE FUNCTION:
  obj = w_lat * (Lat_e2e * 10000 / lat_ub) * 1000
      + w_cost * (Cost * 10000 / cost_max) * 1000
      + w_energy * (Energy * 10000 / energy_ub) * 1000

WHERE:
- Lat_e2e = SUM of shortest-path latencies for ALL application links (end-to-end pipeline)
- Cost = SUM of hostCost[h] for each ACTIVE host (host with >= 1 component placed on it)
- Energy = E_links + E_nodes
  - E_links = SUM over links l: bandwidth[l] * SUM(arc_latency^2 on path of link l)
  - E_nodes = SUM over active hosts h: (200 + cpu_used[h] * 5)

CRITICAL OPTIMIZATION PRINCIPLES (ranked by impact):
1. COLOCATION ELIMINATES COST: Components on the SAME host have 0ms latency AND 0 link energy.
2. BUT E_LINKS CAN DOMINATE: E_links = bw * lat^2 grows QUADRATICALLY with distance.
   When total bandwidth is high, spreading across NEARBY hosts (low lat) beats consolidation
   on a DISTANT host. Compare: 5 hosts all at 2ms = E_links~160 vs 1 host at 40ms = E_links~1.6M.
3. FEWER HOSTS ≠ ALWAYS BETTER: Each host costs only P_static=200 node energy.
   Adding 1 nearby host (+200 energy) is worth it if it saves thousands in link energy.
4. DISTANCE IS SQUARED: Link energy grows with latency^2, so use the NEAREST available hosts.
5. DZ CONSTRAINTS ARE ABSOLUTE: Fixed components MUST stay on their assigned host.
6. CAPACITY IS A HARD WALL: CPU sum <= cpuCap[h] AND RAM sum <= ramCap[h] per host.

KEY INSIGHT: The optimal placement often uses MORE hosts than you'd expect,
as long as those hosts are CLOSE to each other and to DZ hosts.
A 5-host solution with all hosts within 2ms of each other beats a 2-host solution
with hosts 40ms apart.

OPTIMIZATION TRAJECTORY:
Below you will see PREVIOUS placement attempts with their objective scores, sorted from worst to best.
Study the patterns: solutions with LOWER scores use FEWER hosts, COLOCATE more components,
and place free components CLOSE to DZ-constrained hosts.
Generate a NEW placement that achieves a LOWER objective than ALL previous attempts.

REASONING PROTOCOL:
1. Examine the best previous solution — what makes it good?
2. Identify ONE specific change that could improve it (move a component, consolidate hosts)
3. VERIFY the change respects ALL constraints (DZ, CPU, RAM)
4. Compute expected impact: will latency decrease? Will energy decrease?
5. Output the improved placement

HARD CONSTRAINTS (any violation = infeasible = rejected):
- DZ: component must be on its assigned host
- CPU: sum of cpuComp on host h <= cpuCap[h]
- RAM: sum of ramComp on host h <= ramCap[h]
- Connectivity: path must exist between hosts of linked components

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences, no extra text:
{"placement": {"0": host_id, "1": host_id, ...}, "reasoning": "what you changed and why"}"""


# =====================================================================
# A2-A5: ENHANCED LLM SOLVER
# =====================================================================
def _build_problem_description(infra, appli, bounds, w):
    """Construit une description structuree du probleme pour le LLM."""
    lines = []

    # Infrastructure summary (structured, not a wall of text)
    lines.append("=== INFRASTRUCTURE ===")
    lines.append(f"Hosts: {infra.nbH} | Diameter: {infra.diameter}")
    lines.append("")
    lines.append("Host details (id: CPU, RAM, Cost, Tier):")
    for h in range(infra.nbH):
        # Infer tier from capacity
        if infra.cpuCap[h] >= 32:
            tier = "CLOUD"
        elif infra.cpuCap[h] >= 8:
            tier = "FOG"
        else:
            tier = "EDGE"
        lines.append(f"  H{h}: CPU={infra.cpuCap[h]:>3}, RAM={infra.ramCap[h]:>6}, "
                      f"Cost={infra.hostCost[h]:>3}, Tier={tier}")

    # Application DAG
    lines.append("")
    lines.append("=== APPLICATION ===")
    lines.append(f"Components: {appli.nbC} | Links: {appli.nbL}")
    lines.append("")
    lines.append("Component requirements:")
    total_cpu = total_ram = 0
    for c in range(appli.nbC):
        total_cpu += appli.cpuComp[c]; total_ram += appli.ramComp[c]
        lines.append(f"  C{c}: CPU={appli.cpuComp[c]}, RAM={appli.ramComp[c]}")
    lines.append(f"  TOTAL: CPU={total_cpu}, RAM={total_ram}")

    lines.append("")
    lines.append("Application DAG links (source -> dest, bandwidth):")
    for l, (s, t) in enumerate(appli.linkPerServ):
        lines.append(f"  L{l}: C{s} -> C{t}, bw={appli.bdwPair[l]}")

    # DZ constraints
    if appli.DZ:
        lines.append("")
        lines.append("DZ CONSTRAINTS (MANDATORY - cannot be changed):")
        for ci, hi in appli.DZ:
            lines.append(f"  C{ci} MUST be on H{hi}")

    # Colocation analysis (A5: pre-computed sub-problem hints)
    lines.append("")
    lines.append("=== COLOCATION ANALYSIS (pre-computed) ===")
    free_comps = [c for c in range(appli.nbC) if c not in {ci for ci, _ in appli.DZ}]
    free_cpu = sum(appli.cpuComp[c] for c in free_comps)
    free_ram = sum(appli.ramComp[c] for c in free_comps)
    lines.append(f"Free components: {free_comps} (total CPU={free_cpu}, total RAM={free_ram})")

    # For each candidate host, compute EXACT Lat_e2e and Energy if all free go there
    dz_map = {ci: hi for ci, hi in appli.DZ}
    dz_set = set(dz_map.keys())
    candidates = []
    for h in range(infra.nbH):
        # Check capacity
        dz_cpu = sum(appli.cpuComp[ci] for ci, hi in appli.DZ if hi == h)
        dz_ram = sum(appli.ramComp[ci] for ci, hi in appli.DZ if hi == h)
        avail_cpu = infra.cpuCap[h] - dz_cpu
        avail_ram = infra.ramCap[h] - dz_ram
        if avail_cpu < free_cpu or avail_ram < free_ram:
            continue

        # Compute exact Lat_e2e: for each link, if both endpoints on same host -> 0
        # otherwise shortest path between their hosts
        # Build temporary placement: all free on h, DZ as fixed
        tmp_pl = dict(dz_map)
        for c in free_comps:
            tmp_pl[c] = h
        lat_e2e = 0
        energy_link = 0
        for l, (s, t) in enumerate(appli.linkPerServ):
            hs, ht = tmp_pl.get(s, h), tmp_pl.get(t, h)
            if hs == ht:
                pass  # colocated: 0 latency, 0 energy
            else:
                d = infra.sp_latency[hs][ht] if infra.sp_latency[hs][ht] < INF_V else 9999
                lat_e2e += d
                # Energy: bw * sum(arc_lat^2) on path
                path = infra.sp_path.get((hs, ht), [])
                for k in range(len(path) - 1):
                    al = infra.arc_lat.get((path[k], path[k + 1]), 0)
                    energy_link += appli.bdwPair[l] * al * al

        # Active hosts
        active = set(tmp_pl.values())
        cost = sum(infra.hostCost[ah] for ah in active)
        energy_node = sum(P_STATIC + sum(
            appli.cpuComp[c] for c in range(appli.nbC) if tmp_pl[c] == ah
        ) * P_CPU_UNIT for ah in active)
        energy_total = energy_link + energy_node

        candidates.append((h, lat_e2e, cost, energy_total, avail_cpu, avail_ram))

    # Sort by Lat_e2e (primary), then energy (secondary)
    candidates.sort(key=lambda x: (x[1], x[3]))

    lines.append("")
    if candidates:
        lines.append(f"Hosts that can fit ALL free components (sorted BEST first):")
        lines.append(f"  {'Host':>6} {'Lat_e2e':>8} {'Cost':>6} {'Energy':>10} {'CPU_free':>9} {'RAM_free':>9}")
        lines.append(f"  {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*9} {'-'*9}")
        for i, (h, lat, cost, etot, acpu, aram) in enumerate(candidates):
            tag = " <<<< BEST" if i == 0 else ""
            lines.append(f"  H{h:>4} {lat:>7}ms {cost:>6} {etot:>10} {acpu:>9} {aram:>9}{tag}")

        best_h, best_lat = candidates[0][0], candidates[0][1]
        # Show distance breakdown for best candidate
        lines.append(f"")
        lines.append(f"RECOMMENDATION: Place all free components on H{best_h} (Lat_e2e={best_lat}ms)")
        lines.append(f"  Distances from H{best_h} to DZ hosts:")
        for ci, hi in appli.DZ:
            d = infra.sp_latency[best_h][hi] if infra.sp_latency[best_h][hi] < INF_V else 9999
            lines.append(f"    H{best_h} <-> H{hi} (DZ for C{ci}): {d}ms")
        # Show what this means for each link
        lines.append(f"  Link-by-link latencies with this placement:")
        tmp_pl = dict(dz_map)
        for c in free_comps:
            tmp_pl[c] = best_h
        for l, (s, t) in enumerate(appli.linkPerServ):
            hs, ht = tmp_pl[s], tmp_pl[t]
            if hs == ht:
                lines.append(f"    L{l} C{s}->C{t}: H{hs}->H{ht} = 0ms (COLOCATED)")
            else:
                d = infra.sp_latency[hs][ht] if infra.sp_latency[hs][ht] < INF_V else 9999
                lines.append(f"    L{l} C{s}->C{t}: H{hs}->H{ht} = {d}ms")

        if len(candidates) > 1:
            h2 = candidates[1][0]
            lat2 = candidates[1][1]
            lines.append(f"")
            lines.append(f"WARNING: H{h2} is also a candidate but has Lat_e2e={lat2}ms "
                          f"(+{lat2 - best_lat}ms worse than H{best_h}).")

    else:
        lines.append("No single host can fit all free components — must split across hosts.")

    # MULTI-HOST ANALYSIS: show what happens with nearby host clusters
    lines.append("")
    lines.append("=== MULTI-HOST ANALYSIS ===")
    lines.append("Using NEARBY hosts (low mutual latency) often beats single-host consolidation.")
    lines.append("E_links = bw * lat^2: 5 nearby hosts at 2ms each >> 1 distant host at 40ms")
    lines.append("")
    
    # Find best cluster of nearby hosts around DZ hosts
    dz_host_set = set(hi for _, hi in appli.DZ)
    # Score each non-DZ host by average distance to DZ hosts
    host_scores = []
    for h in range(infra.nbH):
        if h in dz_host_set: continue
        avg_dist = 0; n = 0
        for dh in dz_host_set:
            d = infra.sp_latency[h][dh]
            if d < INF_V:
                avg_dist += d; n += 1
        if n > 0:
            host_scores.append((h, avg_dist / n, infra.cpuCap[h], infra.ramCap[h]))
    host_scores.sort(key=lambda x: x[1])  # Nearest to DZ first
    
    if host_scores:
        lines.append("Hosts nearest to DZ hosts (potential placement targets):")
        lines.append(f"  {'Host':>6} {'Avg_dist_DZ':>12} {'CPU':>5} {'RAM':>7}")
        for h, dist, cpu, ram in host_scores[:8]:
            dz_mark = " (DZ)" if h in dz_host_set else ""
            lines.append(f"  H{h:>4} {dist:>11.1f}ms {cpu:>5} {ram:>7}{dz_mark}")
        lines.append(f"  TIP: Try placing free components across these nearby hosts.")
        lines.append(f"  Each new host costs only ~200 node energy, but saves bw*lat^2 link energy.")

    # FURTHER OPTIMIZATION: colocate with DZ hosts (only if we have single-host candidates)
    if candidates:
        lines.append(f"")
        lines.append(f"FURTHER OPTIMIZATION — colocate with DZ hosts:")
        for ci, hi in appli.DZ:
            # Find free components directly linked to ci
            for l, (s, t) in enumerate(appli.linkPerServ):
                neighbor = None
                if s == ci and t in free_comps: neighbor = t
                elif t == ci and s in free_comps: neighbor = s
                if neighbor is None: continue
                # Check if DZ host has capacity for this neighbor
                dz_cpu_on_hi = sum(appli.cpuComp[cj] for cj, hj in appli.DZ if hj == hi)
                dz_ram_on_hi = sum(appli.ramComp[cj] for cj, hj in appli.DZ if hj == hi)
                need_cpu = dz_cpu_on_hi + appli.cpuComp[neighbor]
                need_ram = dz_ram_on_hi + appli.ramComp[neighbor]
                fits = (need_cpu <= infra.cpuCap[hi] and need_ram <= infra.ramCap[hi])
                if fits:
                    # Compute Lat_e2e and Energy with this optimization
                    opt_pl = dict(dz_map)
                    for c in free_comps:
                        opt_pl[c] = best_h
                    opt_pl[neighbor] = hi  # Move neighbor to DZ host
                    opt_lat = 0; opt_elink = 0
                    for ll, (ss, tt) in enumerate(appli.linkPerServ):
                        hss, htt = opt_pl[ss], opt_pl[tt]
                        if hss != htt:
                            d = infra.sp_latency[hss][htt] if infra.sp_latency[hss][htt] < INF_V else 9999
                            opt_lat += d
                            path = infra.sp_path.get((hss, htt), [])
                            for k in range(len(path) - 1):
                                al = infra.arc_lat.get((path[k], path[k + 1]), 0)
                                opt_elink += appli.bdwPair[ll] * al * al
                    lat_saving = best_lat - opt_lat
                    e_saving = candidates[0][3] - opt_elink  # approx
                    if lat_saving > 0 or e_saving > 0:
                        lines.append(
                            f"  -> Move C{neighbor} to H{hi} (with C{ci}): "
                            f"Lat_e2e={opt_lat}ms (save {lat_saving}ms), "
                            f"E_link={opt_elink} (save ~{e_saving})  "
                            f"CPU: {need_cpu}/{infra.cpuCap[hi]}, "
                            f"RAM: {need_ram}/{infra.ramCap[hi]} — FITS!")

    # Latency matrix (compact: only DZ hosts + fog/cloud)
    lines.append("")
    lines.append("=== KEY LATENCIES (ms) ===")
    key_hosts = sorted(set(
        [hi for _, hi in appli.DZ] +
        [h for h in range(infra.nbH) if infra.cpuCap[h] >= 8]  # fog+cloud
    ))
    hdr = "      " + "".join(f"H{h:>3} " for h in key_hosts)
    lines.append(hdr)
    for h1 in key_hosts:
        row = f"  H{h1:>2}: "
        for h2 in key_hosts:
            d = infra.sp_latency[h1][h2]
            row += f"{d:>4} " if d < INF_V else " INF "
        lines.append(row)

    # Objective weights
    lines.append("")
    lines.append(f"=== OBJECTIVE WEIGHTS ===")
    lines.append(f"w_lat={w[0]}, w_cost={w[1]}, w_energy={w[2]}")
    lines.append(f"Bounds: lat_ub={bounds.lat_ub}, cost_max={bounds.cost_max}, "
                  f"energy_ub={bounds.energy_ub}")

    return "\n".join(lines)


def _build_bottleneck_feedback(sol, infra, appli):
    """OPRO-style structured feedback with objective decomposition and actionable suggestions."""
    if sol is None:
        return ""
    lines = []
    lines.append(f"\n=== FEEDBACK ON CURRENT BEST (obj={sol.objective_int}) ===")

    if not sol.feasible:
        lines.append(f"STATUS: INFEASIBLE! Violations: {sol.violations}")
        lines.append(f"FIX THESE FIRST before optimizing the objective.")
        return "\n".join(lines)

    lines.append(f"STATUS: FEASIBLE")
    lines.append(f"  Lat_e2e={sol.total_latency}ms, Cost={sol.infra_cost}, "
                  f"Energy={sol.energy_total} (links={sol.energy_link}, nodes={sol.energy_node})")
    lines.append(f"  Active hosts: {sorted(sol.active_hosts)} ({len(sol.active_hosts)} hosts)")
    lines.append(f"  Placement: {sol.placement}")
    
    # OPRO-style: Show objective decomposition so LLM understands WHERE to optimize
    lines.append(f"\n  OBJECTIVE DECOMPOSITION:")
    # Compute actual normalized contributions
    if hasattr(sol, 'total_latency') and hasattr(sol, 'infra_cost'):
        from math import inf
        lat_ub = 1; cost_max = 1; energy_ub = 1
        # Try to access bounds (they may not be stored on sol)
        try:
            # We'll compute approximate contributions
            lat_contrib = sol.total_latency * 10000 // max(1, 4680) * 340  # w_lat=0.34, approximate
            cost_contrib = sol.infra_cost * 10000 // max(1, 122) * 330
            energy_contrib = sol.energy_total * 10000 // max(1, 15980580) * 330
        except Exception:
            lat_contrib = cost_contrib = energy_contrib = 0
    lines.append(f"    Lat_e2e={sol.total_latency}ms → reduces by colocating or using NEARBY hosts")
    lines.append(f"    Cost={sol.infra_cost} → reduces by using fewer/cheaper hosts")
    lines.append(f"    E_links={sol.energy_link} | E_nodes={sol.energy_node} | E_total={sol.energy_total}")
    
    # KEY: Show which term dominates
    if sol.energy_link > sol.energy_node * 10:
        lines.append(f"    >>> E_links DOMINATES ({sol.energy_link} >> {sol.energy_node}). "
                     f"Reducing inter-host DISTANCES is the #1 priority!")
        lines.append(f"    >>> Consider ADDING nearby hosts rather than consolidating far away.")
    elif sol.energy_link > sol.energy_node:
        lines.append(f"    >>> Link energy > node energy — colocate high-bandwidth pairs!")
    elif len(sol.active_hosts) > 4:
        lines.append(f"    >>> Many hosts ({len(sol.active_hosts)}) — consolidate if possible.")

    # Actionable suggestions
    suggestions = []

    # Suggestion 1: Smart host management based on energy balance
    if sol.energy_link > sol.energy_node * 5 and len(sol.active_hosts) <= 3:
        suggestions.append(
            f"ADD NEARBY HOSTS: E_links={sol.energy_link} DOMINATES E_nodes={sol.energy_node}. "
            f"Adding nearby hosts (even if it adds ~200 node energy each) can save "
            f"thousands in link energy by reducing inter-host distances.")
    elif len(sol.active_hosts) > 5 and sol.energy_node > sol.energy_link:
        suggestions.append(
            f"CONSOLIDATE: You use {len(sol.active_hosts)} hosts but E_nodes={sol.energy_node} > E_links={sol.energy_link}. "
            f"Try reducing host count.")

    # Suggestion 2: Eliminate link energy
    if sol.energy_link > 0:
        # Show which links contribute most energy
        link_energies = []
        for l in range(appli.nbL):
            sC, tC = appli.linkPerServ[l]
            hs, ht = sol.placement[sC], sol.placement[tC]
            if hs != ht:
                path = infra.sp_path.get((hs, ht), [])
                e = 0
                for k in range(len(path) - 1):
                    al = infra.arc_lat.get((path[k], path[k + 1]), 0)
                    e += appli.bdwPair[l] * al * al
                link_energies.append((l, sC, tC, hs, ht, e, sol.consumed_lat.get(l, 0)))
        link_energies.sort(key=lambda x: -x[5])
        if link_energies:
            top = link_energies[0]
            suggestions.append(
                f"TOP ENERGY LINK: L{top[0]} (C{top[1]}->C{top[2]}): H{top[3]}->H{top[4]} "
                f"costs {top[5]} energy + {top[6]}ms latency. "
                f"Colocating C{top[1]} and C{top[2]} on the same host eliminates BOTH.")

    # Suggestion 3: Bottleneck link
    if sol.consumed_lat:
        worst_l = max(sol.consumed_lat, key=lambda l: sol.consumed_lat[l])
        worst_lat = sol.consumed_lat[worst_l]
        if worst_lat > 0:
            sC, tC = appli.linkPerServ[worst_l]
            hs, ht = sol.placement[sC], sol.placement[tC]
            suggestions.append(
                f"BOTTLENECK: L{worst_l} (C{sC}->C{tC}): H{hs}->H{ht} = {worst_lat}ms. "
                f"Colocate C{sC} and C{tC} on the same host to make this 0ms.")

    # Suggestion 4: Per-host capacity usage
    for h in sorted(sol.active_hosts):
        cpu_used = sol.host_cpu_used.get(h, 0)
        cpu_free = infra.cpuCap[h] - cpu_used
        if cpu_free > 4:
            comps_on_h = [c for c in range(appli.nbC) if sol.placement.get(c) == h]
            suggestions.append(
                f"SPARE CAPACITY: H{h} has {cpu_free} CPU free — "
                f"could absorb more components (currently hosts {comps_on_h}).")

    if suggestions:
        lines.append(f"\n  IMPROVEMENT SUGGESTIONS:")
        for i, s in enumerate(suggestions[:5]):
            lines.append(f"  {i+1}. {s}")
    else:
        lines.append(f"\n  This looks near-optimal. Try verifying no single move improves it.")

    return "\n".join(lines)


def _generate_seed_placements(infra, appli, bounds, w):
    """A4: Genere des placements seed diversifies pour guider le LLM."""
    seeds = []
    dz = {ci: hi for ci, hi in appli.DZ}
    free = [c for c in range(appli.nbC) if c not in dz]
    free_cpu = sum(appli.cpuComp[c] for c in free)
    free_ram = sum(appli.ramComp[c] for c in free)

    # Seed 1: Greedy FFD
    seeds.append(("FFD greedy baseline", greedy_placement(infra, appli)))

    # Seed 2: Consolidate on best fog node near DZ
    for target_h in range(infra.nbH):
        dz_cpu = sum(appli.cpuComp[ci] for ci, hi in appli.DZ if hi == target_h)
        dz_ram = sum(appli.ramComp[ci] for ci, hi in appli.DZ if hi == target_h)
        if (infra.cpuCap[target_h] - dz_cpu >= free_cpu and
                infra.ramCap[target_h] - dz_ram >= free_ram):
            # This host can fit all free components
            pl = dict(dz)
            for c in free:
                pl[c] = target_h
            desc = f"All free on H{target_h} ({infra.cpuCap[target_h]} CPU)"
            seeds.append((desc, pl))
            if len(seeds) >= 4:
                break  # Keep top few

    # Seed 3: Consolidate on 2 hosts (split heavy/light)
    dz_host_list = sorted(set(hi for _, hi in appli.DZ))
    if len(dz_host_list) >= 2:
        h_a, h_b = dz_host_list[0], dz_host_list[1]
        # Find best neighbor of each DZ host
        for ha_nb in range(infra.nbH):
            if infra.sp_latency[h_a][ha_nb] >= INF_V: continue
            for hb_nb in range(infra.nbH):
                if infra.sp_latency[h_b][hb_nb] >= INF_V: continue
                if ha_nb == hb_nb: continue
                pl = dict(dz)
                # Split: first half on ha_nb, second half on hb_nb
                half = len(free) // 2
                cu = [0] * infra.nbH; ru = [0] * infra.nbH
                for ci, hi in dz.items():
                    cu[hi] += appli.cpuComp[ci]; ru[hi] += appli.ramComp[ci]
                ok = True
                for i, c in enumerate(free):
                    h = ha_nb if i < half else hb_nb
                    if (cu[h] + appli.cpuComp[c] > infra.cpuCap[h] or
                            ru[h] + appli.ramComp[c] > infra.ramCap[h]):
                        ok = False; break
                    pl[c] = h; cu[h] += appli.cpuComp[c]; ru[h] += appli.ramComp[c]
                if ok and len(pl) == appli.nbC:
                    seeds.append((f"Split: H{ha_nb}+H{hb_nb}", pl))
                    break
            if len(seeds) >= 5:
                break

    # Seed 4: Place on cheapest hosts possible
    pl = dict(dz)
    cu = [0] * infra.nbH; ru = [0] * infra.nbH
    for ci, hi in dz.items():
        cu[hi] += appli.cpuComp[ci]; ru[hi] += appli.ramComp[ci]
    sorted_by_cost = sorted(range(infra.nbH), key=lambda h: infra.hostCost[h])
    for c in sorted(free, key=lambda c: -(appli.cpuComp[c] + appli.ramComp[c])):
        for h in sorted_by_cost:
            if (cu[h] + appli.cpuComp[c] <= infra.cpuCap[h] and
                    ru[h] + appli.ramComp[c] <= infra.ramCap[h]):
                pl[c] = h; cu[h] += appli.cpuComp[c]; ru[h] += appli.ramComp[c]
                break
        else:
            pl[c] = 0
    seeds.append(("Cheapest hosts first", pl))

    # Seed 5: Random compact (pick random fog node)
    fog_nodes = [h for h in range(infra.nbH) if 8 <= infra.cpuCap[h] <= 16]
    if fog_nodes:
        rh = random.choice(fog_nodes)
        pl = dict(dz); cu = [0] * infra.nbH; ru = [0] * infra.nbH
        for ci, hi in dz.items():
            cu[hi] += appli.cpuComp[ci]; ru[hi] += appli.ramComp[ci]
        for c in free:
            if (cu[rh] + appli.cpuComp[c] <= infra.cpuCap[rh] and
                    ru[rh] + appli.ramComp[c] <= infra.ramCap[rh]):
                pl[c] = rh; cu[rh] += appli.cpuComp[c]; ru[rh] += appli.ramComp[c]
            else:
                # Overflow: nearest host with capacity
                for h2 in sorted(range(infra.nbH),
                                  key=lambda h: infra.sp_latency[rh][h]
                                  if infra.sp_latency[rh][h] < INF_V else 9999):
                    if (cu[h2] + appli.cpuComp[c] <= infra.cpuCap[h2] and
                            ru[h2] + appli.ramComp[c] <= infra.ramCap[h2]):
                        pl[c] = h2; cu[h2] += appli.cpuComp[c]; ru[h2] += appli.ramComp[c]
                        break
                else:
                    pl[c] = 0
        seeds.append((f"Random fog H{rh}", pl))

    # Seed 6: Nearest-neighbor greedy (place each free component on host nearest to its placed neighbors)
    # This generates the multi-host spread solutions that CSP often finds optimal
    pl = dict(dz)
    cu = [0] * infra.nbH; ru = [0] * infra.nbH
    for ci, hi in dz.items():
        cu[hi] += appli.cpuComp[ci]; ru[hi] += appli.ramComp[ci]
    # Build adjacency for free components
    comp_links = defaultdict(list)
    for l, (s, t) in enumerate(appli.linkPerServ):
        comp_links[s].append((t, appli.bdwPair[l]))
        comp_links[t].append((s, appli.bdwPair[l]))
    # Order: process components with most links to already-placed components first
    placed = set(dz.keys())
    remaining = list(free)
    while remaining:
        # Score each remaining component by connectivity to placed components
        best_c, best_h, best_score = None, None, float("inf")
        for c in remaining:
            for h in range(infra.nbH):
                if cu[h] + appli.cpuComp[c] > infra.cpuCap[h]: continue
                if ru[h] + appli.ramComp[c] > infra.ramCap[h]: continue
                sc = 0
                for nb, bw in comp_links[c]:
                    if nb in placed:
                        nh = pl[nb]
                        d = infra.sp_latency[h][nh] if infra.sp_latency[h][nh] < INF_V else 9999
                        sc += bw * d * d  # Minimize link energy
                # Prefer already-active hosts (no extra node cost)
                if cu[h] == 0 and h not in set(dz.values()):
                    sc += 200  # P_STATIC penalty for new host
                if sc < best_score:
                    best_score = sc; best_c = c; best_h = h
        if best_c is not None:
            pl[best_c] = best_h
            cu[best_h] += appli.cpuComp[best_c]; ru[best_h] += appli.ramComp[best_c]
            placed.add(best_c)
            remaining.remove(best_c)
        else:
            # Fallback: place on any host with capacity
            c = remaining.pop(0)
            for h in range(infra.nbH):
                if cu[h] + appli.cpuComp[c] <= infra.cpuCap[h] and ru[h] + appli.ramComp[c] <= infra.ramCap[h]:
                    pl[c] = h; cu[h] += appli.cpuComp[c]; ru[h] += appli.ramComp[c]; placed.add(c); break
            else:
                pl[c] = 0; placed.add(c)
    if len(pl) == appli.nbC:
        seeds.append(("NN-greedy (min link energy)", pl))

    # Seed 7: DZ-spread (place each free component on its closest DZ host with capacity)
    pl = dict(dz)
    cu = [0] * infra.nbH; ru = [0] * infra.nbH
    for ci, hi in dz.items():
        cu[hi] += appli.cpuComp[ci]; ru[hi] += appli.ramComp[ci]
    dz_hosts_list = sorted(set(hi for _, hi in appli.DZ))
    # For each free component, find which DZ host it links to most, place on/near it
    for c in free:
        # Score each DZ host by total bandwidth to this component
        dz_scores = []
        for dh in dz_hosts_list:
            bw_sum = 0
            for nb, bw in comp_links[c]:
                if nb in dz and dz[nb] == dh:
                    bw_sum += bw
            dz_scores.append((dh, bw_sum))
        dz_scores.sort(key=lambda x: -x[1])  # Highest bandwidth first
        placed_ok = False
        for target_dh, _ in dz_scores:
            # Try target DZ host first, then nearest hosts to it
            candidates = sorted(range(infra.nbH),
                                key=lambda h: infra.sp_latency[target_dh][h] if infra.sp_latency[target_dh][h] < INF_V else 9999)
            for h in candidates[:6]:
                if cu[h] + appli.cpuComp[c] <= infra.cpuCap[h] and ru[h] + appli.ramComp[c] <= infra.ramCap[h]:
                    pl[c] = h; cu[h] += appli.cpuComp[c]; ru[h] += appli.ramComp[c]
                    placed_ok = True; break
            if placed_ok: break
        if not placed_ok:
            pl[c] = 0  # fallback
    if len(pl) == appli.nbC:
        seeds.append(("DZ-spread (near DZ hosts)", pl))

    return seeds


def _build_opro_trajectory(trajectory, max_shown=20):
    """OPRO-style: Build compact optimization trajectory sorted ascending (worst→best).
    Compact format reduces tokens by ~60% vs verbose format."""
    if not trajectory:
        return ""
    # Sort ascending (worst first, best last — OPRO finding: recency bias helps)
    shown = sorted(trajectory, key=lambda x: -x[0])[-max_shown:]  # keep best N
    shown.sort(key=lambda x: x[0], reverse=True)  # worst first
    
    lines = ["\n=== TRAJECTORY (worst→best, LOWER=BETTER) ==="]
    for i, (score, pl, reasoning) in enumerate(shown):
        active = len(set(pl.values()))
        # Compact: one line per attempt
        pl_compact = ",".join(f"{c}:{h}" for c, h in sorted(pl.items()))
        insight = f" | {reasoning[:80]}" if reasoning else ""
        lines.append(f"  #{i+1} score={score} hosts={active} [{pl_compact}]{insight}")
    
    # OPRO meta-instruction (compact)
    best_score = shown[-1][0] if shown else float("inf")
    lines.append(f"\nBEST={best_score}. Generate NEW placement with score < {best_score}.")
    return "\n".join(lines)


def _build_opro_prompt(problem_desc, trajectory, feedback, best, infra, appli,
                       iteration, max_iter, temperature, strategy_hint=""):
    """Build OPRO-style meta-prompt — compact version for faster inference."""
    parts = [problem_desc]
    
    # OPRO trajectory (core innovation)
    traj_text = _build_opro_trajectory(trajectory)
    if traj_text:
        parts.append(traj_text)
    
    # Structured feedback (only if meaningful)
    if feedback and best:
        parts.append(feedback)
    
    # Best known (compact)
    if best and best.feasible:
        parts.append(f"\n--- BEAT THIS: score={best.objective_int} "
                     f"hosts={sorted(best.active_hosts)} "
                     f"lat={best.total_latency}ms E={best.energy_total} ---")
        parts.append(f"Placement: {best.placement}")
    
    # Strategy hint
    if strategy_hint:
        parts.append(f"\nHINT[{iteration}/{max_iter}]: {strategy_hint}")
    
    parts.append(f'\nRespond JSON: {{"placement": {{"0": host_id, ...}}, "reasoning": "..."}}')
    
    return "\n".join(parts)


def run_llm(infra, appli, bounds, w, provider, model, max_iter=5, verbose=True):
    """OPRO-inspired LLM optimizer with optimization trajectory and multi-solution generation."""
    total_tok_in = 0; total_tok_out = 0; total_llm_time = 0.0
    best = None; t0 = time.time()
    
    # OPRO trajectory: list of (score, placement_dict, reasoning_str)
    trajectory = []

    # --- Phase 1: Seed evaluation (A2+A4+A7+A8) ---
    seeds = _generate_seed_placements(infra, appli, bounds, w)
    if verbose:
        print(f"    Seeds generees: {len(seeds)}")
    for name, pl in seeds:
        sol = evaluate_solution(pl, infra, appli, bounds, *w)
        # Add to OPRO trajectory
        trajectory.append((sol.objective_int if sol.feasible else 10**9,
                          dict(sol.placement), name))
        if sol.feasible and (best is None or sol.objective < best.objective):
            best = sol
            if verbose:
                print(f"      {name}: obj={sol.objective_int} "
                      f"lat={sol.total_latency}ms E={sol.energy_total} *BEST*")
        elif verbose:
            tag = "OK" if sol.feasible else "FAIL"
            print(f"      {name}: obj={sol.objective_int} [{tag}]")

    # --- Phase 2: Build problem description ---
    problem_desc = _build_problem_description(infra, appli, bounds, w)

    # --- Phase 3: OPRO-style iterative optimization ---
    # OPRO key parameters (from paper Section 5.3):
    # - 8 solutions per step (we use multi-call)
    # - Temperature 1.0 default (we adapt: low→high)
    # - Top 20 solutions in trajectory
    # - 3 exemplars
    # - Ascending order (worst→best) for recency bias
    
    opro_strategies = [
        # Direct exploitation (follow best)
        "EXPLOIT: The best solution is shown above. Make ONE small change to improve it. "
        "Move a single component to reduce latency or energy. Keep what works.",
        
        # Nearby spread (KEY: counteracts consolidation bias)
        "NEARBY SPREAD: Place each free component on the NEAREST host to its DZ-linked neighbors. "
        "Using 4-5 nearby hosts (all within 2-5ms) is often BETTER than 1-2 distant hosts. "
        "Check the KEY LATENCIES matrix to find hosts that are close to EACH OTHER and to DZ hosts. "
        "E_links = bw * lat^2, so 5 hosts at 2ms = ~20 energy per link vs 1 host at 40ms = ~1600.",
        
        # Bottleneck fix
        "FIX BOTTLENECK: Look at the FEEDBACK. The worst link contributes the most to latency "
        "and energy (quadratic!). Colocate its endpoints on the same host, OR move them to "
        "two hosts that are VERY close (1-2ms apart).",
        
        # DZ-centric placement
        "DZ-CENTRIC: For each free component, find which DZ host it communicates with most "
        "(highest bandwidth links). Place it ON that DZ host if capacity allows, "
        "or on the NEAREST host to that DZ host. This minimizes the dominant E_links term.",
        
        # Energy-first (FIXED: don't say minimize hosts FIRST)
        "ENERGY MINIMIZER: E_links = bw * lat^2 is usually the DOMINANT energy term. "
        "Focus on minimizing inter-host DISTANCES rather than host count. "
        "Check: is your current E_links >> E_nodes? If yes, add nearby hosts to reduce distances.",
        
        # Cost-aware
        "COST REDUCER: Check hostCost values. Some hosts are much cheaper. "
        "Can you achieve similar latency on a cheaper set of hosts?",
        
        # Creative exploration (high diversity)
        "CREATIVE: Try a placement VERY DIFFERENT from all previous attempts. "
        "If all attempts used 2-3 hosts, try 5-6 NEARBY hosts. "
        "If all used distant hosts, try colocating more. Break the pattern.",
        
        # Verify and micro-optimize
        "MICRO-OPTIMIZE: For EACH component in the best solution, ask: "
        "'would moving this component to any other host improve the objective?' "
        "Check every possible single-component move systematically.",
    ]

    # OPRO: generate solutions per step — fewer in exploit phases, more in explore
    effective_steps = max(1, max_iter)
    stale_iterations = 0  # Track iterations without improvement for early convergence
    max_stale = 3  # Allow more exploration before stopping
    
    for it in range(1, effective_steps + 1):
        # Early convergence: stop if no improvement for max_stale iterations
        if stale_iterations >= max_stale:
            if verbose:
                print(f"    Convergence detectee apres {it-1} iterations (stale={stale_iterations})")
            break
        
        strat_idx = (it - 1) % len(opro_strategies)
        strat = opro_strategies[strat_idx]
        strat_short = strat[:60] + "..."

        # OPRO-style temperature schedule
        if it <= effective_steps // 3:
            temperature = 0.15  # Exploit phase
        elif it <= 2 * effective_steps // 3:
            temperature = 0.5 + (it - effective_steps // 3) * 0.1  # Explore phase
        else:
            temperature = 0.2  # Final exploit
        temperature = min(temperature, 1.0)
        
        # Adaptive solutions per step: 1 in exploit, 2 in explore
        is_explore = effective_steps // 3 < it <= 2 * effective_steps // 3
        solutions_per_step = 2 if is_explore else 1

        # Build OPRO meta-prompt
        feedback = _build_bottleneck_feedback(best, infra, appli)
        prompt = _build_opro_prompt(
            problem_desc, trajectory, feedback, best, infra, appli,
            it, effective_steps, temperature, strat)

        if verbose: print(f"    It {it}/{effective_steps}: {strat_short}", end=" ")

        # OPRO: Generate multiple solutions per step (paper: 8 per step)
        step_solutions = []
        for attempt in range(solutions_per_step):
            try:
                t_call = time.time()
                # Vary temperature slightly between attempts for diversity
                t_var = temperature + attempt * 0.15
                text, ti, to = _call_llm_with_tokens(
                    prompt, LLM_SYSTEM, provider, model, temperature=min(t_var, 1.2))
                dt = time.time() - t_call
                total_tok_in += ti; total_tok_out += to; total_llm_time += dt
            except Exception as e:
                if verbose and attempt == 0: print(f"ERREUR: {e}")
                continue

            pl = _parse_llm_json(text, appli.nbC)
            if pl is None:
                continue

            sol = evaluate_solution(pl, infra, appli, bounds, *w)
            if not sol.feasible:
                pl = repair_placement(pl, infra, appli)
                sol = evaluate_solution(pl, infra, appli, bounds, *w)

            # Extract reasoning from LLM response
            reasoning = ""
            try:
                m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
                if m: reasoning = m.group(1)[:120]
            except Exception:
                pass

            step_solutions.append((sol, reasoning))
            
            # Add to OPRO trajectory
            trajectory.append((
                sol.objective_int if sol.feasible else 10**9,
                dict(sol.placement),
                reasoning or f"LLM iter {it} attempt {attempt+1}"
            ))

        # Evaluate step solutions
        step_improved = False
        for sol, reasoning in step_solutions:
            is_best = (best is None or (sol.feasible and
                       (not best.feasible or sol.objective < best.objective)))
            if is_best:
                best = sol
                step_improved = True
        
        # Track convergence
        if step_improved:
            stale_iterations = 0
        else:
            stale_iterations += 1

        if verbose and step_solutions:
            sol0, _ = step_solutions[0]
            tag = "OK" if sol0.feasible else "FAIL"
            n_ok = sum(1 for s, _ in step_solutions if s.feasible)
            star = " *BEST*" if any(
                s.feasible and (best is None or s.objective_int == best.objective_int)
                for s, _ in step_solutions) else ""
            tok = total_tok_in + total_tok_out
            print(f"[{tag}] {len(step_solutions)} sols, {n_ok} OK, "
                  f"best_step={min(s.objective_int for s,_ in step_solutions)} "
                  f"tok={tok}{star}")
        elif verbose:
            print("no valid solutions")

    # --- Phase 4: Post-LLM local search ---
    if best and best.feasible:
        post_best = standalone_local_search(best.placement, infra, appli, bounds, w, max_iter=150)
        if post_best.feasible and post_best.objective < best.objective:
            best = post_best
        if verbose:
            print(f"    Post-RL: obj={best.objective_int}")

    t_total = time.time() - t0
    if best is None:
        best = evaluate_solution(greedy_placement(infra, appli), infra, appli, bounds, *w)
    best.solve_time = t_total

    # E_solver
    total_tok = total_tok_in + total_tok_out
    e_per_1k = ENERGY_PER_1K_TOKENS.get(provider, 0.004)
    cost_in, cost_out = COST_PER_1K_TOKENS.get(provider, (0, 0))
    if provider == "ollama":
        e_solver = P_CPU_LOCAL * total_llm_time / 3600.0
        e_method = f"P*T ({P_CPU_LOCAL:.0f}W)"
    else:
        e_solver = total_tok / 1000.0 * e_per_1k
        e_method = f"tok*{e_per_1k}Wh/1k"
    cost_usd = total_tok_in / 1000.0 * cost_in + total_tok_out / 1000.0 * cost_out

    if verbose:
        print(f"  [LLM] Tokens cumul: {total_tok_in}in + {total_tok_out}out = {total_tok}")
        print(f"  [LLM] E_solver={e_solver:.6f}Wh | ${cost_usd:.4f}")

    return SolverResult(
        name=f"LLM ({model})", sol=best,
        t_total_s=t_total, tokens_in=total_tok_in, tokens_out=total_tok_out,
        e_solver_wh=e_solver, cost_usd=cost_usd, e_method=e_method)


# =====================================================================
# SELECTEUR DE PROVIDER LLM
# =====================================================================
def _ollama_list_models():
    """Liste les modeles Ollama disponibles."""
    import urllib.request, urllib.error
    urls = [os.environ.get("OLLAMA_HOST", "http://localhost:11434")]
    if "localhost" in urls[0]:
        urls.append(urls[0].replace("localhost", "127.0.0.1"))
    for url in urls:
        try:
            req = urllib.request.Request(f"{url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                if models:
                    os.environ["OLLAMA_HOST"] = url
                    return models
        except Exception:
            pass
    return []


def _auto_detect_provider():
    """v12-style auto-detection: try Gemini, Anthropic, OpenAI, Ollama in order."""
    # 1. Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if gemini_key:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        return "gemini", model
    # 2. Anthropic
    if os.environ.get("ANTHROPIC_API_KEY", ""):
        return "anthropic", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    # 3. OpenAI
    if os.environ.get("OPENAI_API_KEY", ""):
        return "openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    # 4. Ollama
    models = _ollama_list_models()
    if models:
        for pref in ["gemma2", "llama3", "mistral", "qwen2"]:
            for m in models:
                if pref in m: return "ollama", m
        return "ollama", models[0]
    return None, None


def select_llm_provider():
    providers = {
        "1": ("anthropic", "claude-sonnet-4-20250514"),
        "2": ("openai", "gpt-4o"),
        "3": ("gemini", "gemini-2.0-flash"),
        "4": ("ollama", "llama3"),
        "5": ("auto", "auto"),
    }
    print("\n  LLM Provider:")
    print("    1. Anthropic (Claude)")
    print("    2. OpenAI (GPT-4o)")
    print("    3. Google (Gemini)")
    print("    4. Ollama (local)")
    print("    5. Auto-detect (v12-style)")
    while True:
        ch = input("  Choix [1-5]: ").strip()
        if ch in providers: break
    prov, default_mod = providers[ch]

    if prov == "auto":
        prov, mod = _auto_detect_provider()
        if prov:
            print(f"  Auto-detected: {prov} ({mod})")
            return prov, mod
        else:
            print("  Aucun LLM detecte. Utilisez --skip-llm ou configurez un provider.")
            return "ollama", "llama3"

    if prov == "ollama":
        print("  Detection des modeles Ollama...")
        models = _ollama_list_models()
        if models:
            print(f"  Modeles disponibles:")
            for i, m in enumerate(models, 1):
                print(f"    {i}. {m}")
            sel = input(f"  Choix [1-{len(models)}] ou nom [{default_mod}]: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(models):
                return prov, models[int(sel) - 1]
            elif sel:
                return prov, sel
            return prov, default_mod
        else:
            print("  Ollama non detecte, modele par defaut.")
            custom = input(f"  Nom du modele [{default_mod}]: ").strip()
            return prov, custom if custom and not custom.isdigit() else default_mod
    else:
        custom = input(f"  Modele [{default_mod}]: ").strip()
        if custom and not custom.isdigit():
            return prov, custom
        return prov, default_mod


# =====================================================================
# RAPPORT FINAL COMPARATIF
# =====================================================================
def print_comparison(results: List[SolverResult], bounds, infra, appli):
    """Affiche le grand tableau comparatif + ROI."""
    valid = [r for r in results if r is not None]
    if not valid:
        print("Aucun resultat."); return

    W = 24
    N = len(valid)

    print(f"\n{'#' * (30 + (W + 3) * N)}")
    print(f"{'#':2} {'TABLEAU COMPARATIF COMPLET':^{28 + (W + 3) * N - 4}} {'#':>2}")
    print(f"{'#' * (30 + (W + 3) * N)}")
    print(f" Lat_ub={bounds.lat_ub} | Cost_max={bounds.cost_max} | E_ub={bounds.energy_ub}")
    print(f" Latence = SOMME bout-en-bout (pipeline serie)")

    hdr = f" {'':30}"
    for r in valid: hdr += f"| {r.name:>{W}} "
    print(f"\n{hdr}")
    sep = f" {'-' * 30}" + (f"+-{'-' * W}-") * N
    print(sep)

    def row(label, fn, fmt="{}", unit=""):
        line = f" {label:<30}"
        for r in valid:
            v = fn(r)
            line += f"| {(fmt.format(v) + unit):>{W}} "
        print(line)

    # --- METRIQUES PLACEMENT ---
    print(f" {'METRIQUES PLACEMENT':^30}" + f"  {'':>{W}} " * N)
    row("Statut", lambda r: "FAISABLE" if r.sol.feasible else "INFAISABLE")
    row("Lat e2e (somme)", lambda r: r.sol.total_latency, "{}", " ms")
    row("Lat pire lien (max)", lambda r: r.sol.worst_case_latency, "{}", " ms")
    row("Cout deploiement", lambda r: r.sol.infra_cost)
    row("Energie liens", lambda r: r.sol.energy_link)
    row("Energie noeuds", lambda r: r.sol.energy_node)
    row("Energie TOTALE", lambda r: r.sol.energy_total)
    row("Hotes actifs", lambda r: len(r.sol.active_hosts))
    row("Objectif (int)", lambda r: r.sol.objective_int)
    print(sep)

    # --- COUT DU SOLVEUR ---
    print(f" {'COUT DU SOLVEUR':^30}" + f"  {'':>{W}} " * N)
    row("Temps total", lambda r: r.t_total_s, "{:.3f}", " s")
    row("Tokens (in+out)", lambda r: r.tokens_in + r.tokens_out)
    row("E_solver (Wh)", lambda r: r.e_solver_wh, "{:.6f}")
    row("E_solver (J)", lambda r: r.e_solver_wh * 3600, "{:.3f}")
    row("Cout ($)", lambda r: r.cost_usd, "{:.4f}")
    row("Methode E_solver", lambda r: r.e_method)
    print(sep)

    # --- ROI ENERGETIQUE ---
    # Baseline = FFD (premier resultat nommé "Glouton")
    ffd = next((r for r in valid if "Glouton" in r.name or "FFD" in r.name), valid[-1])
    e_ffd = ffd.sol.energy_total

    print(f" {'ROI ENERGETIQUE (vs FFD)':^30}" + f"  {'':>{W}} " * N)
    row("Delta E (gain placement)",
        lambda r: e_ffd - r.sol.energy_total if r.sol.feasible else "N/A")

    def roi_str(r):
        if not r.sol.feasible: return "N/A"
        delta = e_ffd - r.sol.energy_total
        if r.e_solver_wh <= 0: return "inf" if delta > 0 else "baseline"
        # Note: delta en unites modele, E_solver en Wh - ratio indicatif
        return f"{delta / r.e_solver_wh:.1f}" if r.e_solver_wh > 1e-10 else "~inf"
    row("ROI_E (Delta/E_solver)", roi_str)

    def payback(r):
        if not r.sol.feasible or "Glouton" in r.name or "FFD" in r.name: return "-"
        delta = e_ffd - r.sol.energy_total
        if delta <= 0: return "jamais"
        if r.e_solver_wh <= 1e-10: return "~0s"
        # T* = E_solver / delta * T_normalisation
        # En unites modele, pas directement en Wh, donc T* indicatif
        return f"{r.e_solver_wh / delta * 3600:.1f}s"
    row("T* (payback)", payback)
    print(sep)

    # --- PLACEMENTS ---
    print(f"\n PLACEMENTS:")
    for r in valid:
        print(f"   {r.name:35s}: {r.sol.placement}")

    # --- EXPORT JSON ---
    export = {}
    for r in valid:
        export[r.name] = {
            "placement": {str(k): v for k, v in r.sol.placement.items()},
            "metrics": {
                "lat_e2e_ms": r.sol.total_latency,
                "lat_max_ms": r.sol.worst_case_latency,
                "cout": r.sol.infra_cost,
                "energie_totale": r.sol.energy_total,
                "objectif": r.sol.objective_int,
                "feasible": r.sol.feasible,
            },
            "solver_cost": {
                "t_total_s": round(r.t_total_s, 4),
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "e_solver_wh": round(r.e_solver_wh, 8),
                "cost_usd": round(r.cost_usd, 4),
                "method": r.e_method,
            }
        }
    json_path = "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"\n  Export JSON: {json_path}")
    print(f"{'#' * (30 + (W + 3) * N)}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    pa = argparse.ArgumentParser(description="TPDS2 Benchmark Unifie: CSP / LLM / Glouton (FFD)")
    # pa.add_argument("--infra", default="properties/Infra_8nodes.properties")
    # pa.add_argument("--appli", default="properties/Appli_4comps.properties")
    pa.add_argument("--infra", default="properties/Infra_16nodes_fog3tier.properties")
    pa.add_argument("--appli", default="properties/Appli_8comps_smartbuilding.properties")
    #pa.add_argument("--infra", default="properties/Infra_24nodes_dcns.properties")
    #pa.add_argument("--appli", default="properties/Appli_10comps_dcns.properties")
    #pa.add_argument("--infra", default="properties/Infra_28nodes_smartcity.properties")
    #pa.add_argument("--appli", default="properties/Appli_11comps_smartcity.properties")
    #pa.add_argument("--infra", default="properties/Infra_32nodes_hospital.properties")
    #pa.add_argument("--appli", default="properties/Appli_12comps_ehealth.properties")
    
    
    pa.add_argument("--w-lat", type=float, default=0.)
    pa.add_argument("--w-cost", type=float, default=0.)
    pa.add_argument("--w-energy", type=float, default=1)
    pa.add_argument("--llm-iter", type=int, default=8)
    pa.add_argument("--skip-csp", action="store_true", help="Ne pas lancer CP-SAT")
    pa.add_argument("--skip-llm", action="store_true", help="Ne pas lancer le LLM")
    pa.add_argument("--provider", default=None, help="LLM provider: anthropic/openai/gemini/ollama")
    pa.add_argument("--model", default=None, help="Nom du modele LLM")
    args = pa.parse_args()
    w = (args.w_lat, args.w_cost, args.w_energy)

    print("=" * 80)
    print(" TPDS2 - BENCHMARK UNIFIE")
    print(" CSP (CP-SAT) | LLM | Glouton (FFD)")
    print("=" * 80)

    # --- Chargement ---
    print("\n[1/4] Chargement des donnees...")
    infra = load_infra(args.infra)
    appli = load_appli(args.appli)
    bounds = compute_norm_bounds(infra, appli)
    print(f"  Infra: {infra.nbH} hotes | Diametre: {infra.diameter}")
    print(f"  Appli: {appli.nbC} composants | {appli.nbL} liens | {appli.nbDZ} DZ")
    print(f"  Poids: w_lat={w[0]} w_cost={w[1]} w_energy={w[2]}")
    print(f"  Bornes: lat_ub={bounds.lat_ub} cost_max={bounds.cost_max} energy_ub={bounds.energy_ub}")

    results = []

    # --- 1. GLOUTON (FFD) ---
    print("\n[2/4] Glouton (FFD)...")
    t0 = time.time()
    g_sol = evaluate_solution(greedy_placement(infra, appli), infra, appli, bounds, *w)
    t_ffd = time.time() - t0
    g_sol.solve_time = t_ffd
    print(f"  Obj={g_sol.objective_int} Lat_e2e={g_sol.total_latency}ms "
          f"E={g_sol.energy_total} t={t_ffd:.4f}s")
    results.append(SolverResult(
        name="Glouton (FFD)", sol=g_sol,
        t_total_s=t_ffd, e_solver_wh=P_CPU_LOCAL * t_ffd / 3600.0,
        e_method="~0 (neg.)"))

    # --- 2. CSP ---
    if not args.skip_csp:
        print("\n[3/4] CSP (CP-SAT)...")
        csp_result = run_csp(args.infra, args.appli, w, bounds, infra, appli)
        if csp_result:
            results.append(csp_result)
    else:
        print("\n[3/4] CSP: SKIP (--skip-csp)")

    # --- 3. LLM ---
    if not args.skip_llm:
        print("\n[4/4] LLM...")
        if args.provider and args.model:
            provider, model = args.provider, args.model
        elif args.provider == "auto":
            provider, model = _auto_detect_provider()
            if not provider:
                print("  Aucun LLM detecte, skip LLM."); results.append(None)
                provider = None
        elif args.provider:
            defaults = {"anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o",
                        "gemini": "gemini-2.0-flash", "ollama": "llama3"}
            provider = args.provider; model = defaults.get(provider, "gpt-4o")
        else:
            provider, model = select_llm_provider()
        print(f"  Provider: {provider} | Modele: {model}")
        llm_result = run_llm(infra, appli, bounds, w, provider, model,
                             max_iter=args.llm_iter, verbose=True)
        results.append(llm_result)
    else:
        print("\n[4/4] LLM: SKIP (--skip-llm)")

    # --- RAPPORT ---
    print_comparison(results, bounds, infra, appli)


if __name__ == "__main__":
    main()
