

import os
import re
import json
import time
import random
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass

import networkx as nx

from src.placementAlgo import PlacementAlgo, PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph

# =====================================================================
# CONSTANTS
# =====================================================================
P_STATIC = 200
P_CPU_UNIT = 5
NORM = 10000
INF_V = 10 ** 9
P_CPU_LOCAL = float(os.environ.get("P_CPU_WATTS", "45"))  # Watts

# Energy per 1k tokens (Wh) - source: Luccioni et al. 2023
ENERGY_PER_1K_TOKENS = {
    "anthropic": 0.004,
    "openai": 0.004,
    "gemini": 0.003,
    "ollama": 0.0,  # P*T measurement
}

# Cost USD per 1k tokens (input, output)
COST_PER_1K_TOKENS = {
    "anthropic": (0.003, 0.015),
    "openai": (0.005, 0.015),
    "gemini": (0.0005, 0.0015),
    "ollama": (0.0, 0.0),
}

# =====================================================================
# LLM SYSTEM PROMPT
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
# HELPER DATACLASSES
# =====================================================================
@dataclass
class NormBounds:
    """Normalization bounds for objective computation"""
    lat_ub: float = 1000.0
    cost_max: float = 1000.0
    energy_ub: float = 100000.0


@dataclass
class PlacementSolution:
    """Internal solution representation with metrics"""
    placement: Dict[int, int]
    feasible: bool = True
    violations: List[str] = None
    total_latency: float = 0.0
    worst_case_latency: float = 0.0
    infra_cost: int = 0
    energy_link: float = 0.0
    energy_node: float = 0.0
    energy_total: float = 0.0
    objective: float = float('inf')
    objective_int: int = 10**9
    active_hosts: set = None
    host_cpu_used: Dict[int, int] = None
    host_ram_used: Dict[int, int] = None
    paths: Dict[int, List[int]] = None
    consumed_lat: Dict[int, float] = None
    solve_time: float = 0.0
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.active_hosts is None:
            self.active_hosts = set()
        if self.host_cpu_used is None:
            self.host_cpu_used = {}
        if self.host_ram_used is None:
            self.host_ram_used = {}
        if self.paths is None:
            self.paths = {}
        if self.consumed_lat is None:
            self.consumed_lat = {}


# =====================================================================
# LLM COMMUNICATION HELPERS
# =====================================================================
def _estimate_tokens(text: str) -> int:
    """Rough token estimation"""
    return max(1, len(text) // 4)


def _call_llm_with_tokens(prompt: str, system: str, provider: str, 
                          model: str, temperature: float = 0.3) -> Tuple[str, int, int]:
    """Call LLM and return (text, tokens_in, tokens_out)"""
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
        try:
            from google import genai
            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", "") or 
                                 os.environ.get("GEMINI_API_KEY", ""))
            r = client.models.generate_content(
                model=model, contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system, max_output_tokens=4096, temperature=temperature))
            ti = getattr(r.usage_metadata, 'prompt_token_count', None) or _estimate_tokens(system + prompt)
            to = getattr(r.usage_metadata, 'candidates_token_count', None) or _estimate_tokens(r.text)
            return r.text, ti, to
        except ImportError:
            # REST fallback
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
                    if e.code == 404 and api_version == "v1beta":
                        continue
                    raise
            raise ValueError(f"Gemini API failed for model {model}")

    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False, 
            "options": {"temperature": temperature, "num_predict": 4096}
        }).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        ti = data.get("prompt_eval_count", _estimate_tokens(system + prompt))
        to = data.get("eval_count", _estimate_tokens(data["message"]["content"]))
        return data["message"]["content"], ti, to

    raise ValueError(f"Unknown provider: {provider}")


def _parse_llm_json(response: str, nbC: int) -> Optional[Dict[int, int]]:
    """Extract placement dict from LLM response"""
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
# EVALUATION & METRICS
# =====================================================================
def _compute_norm_bounds(network_graph: NetworkGraph, service_graph: ServiceGraph) -> NormBounds:
    """Compute normalization bounds for objective function"""
    bounds = NormBounds()
    
    NG = network_graph.G
    SG = service_graph.G
    
    # Get diameter
    diameter = network_graph.metadata.get('network.diameter', len(NG.nodes()))
    
    # Max arc latency
    max_arc_lat = max((d.get('latency', 1) for _, _, d in NG.edges(data=True)), default=1)
    
    # Latency upper bound
    nbL = SG.number_of_edges()
    bounds.lat_ub = max(1, nbL * diameter * max_arc_lat)
    
    # Cost upper bound (assuming all hosts may have a cost attribute, default to CPU)
    total_cost = sum(
        NG.nodes[h].get('cost', NG.nodes[h].get('cpu', 1)) 
        for h in NG.nodes()
    )
    bounds.cost_max = max(1, total_cost)
    
    # Energy bounds
    max_flow = max((d.get('bandwidth', 0) for _, _, d in SG.edges(data=True)), default=1)
    bounds.e_link_ub = max(1, max_flow * max_arc_lat * max_arc_lat)
    
    max_cpu = max((NG.nodes[h].get('cpu', 0) for h in NG.nodes()), default=1)
    bounds.e_node_ub = len(NG.nodes()) * (P_STATIC + max_cpu * P_CPU_UNIT)
    bounds.energy_ub = bounds.e_link_ub + bounds.e_node_ub
    
    return bounds


def _evaluate_solution(placement: Dict[int, int], 
                       network_graph: NetworkGraph,
                       service_graph: ServiceGraph,
                       bounds: NormBounds,
                       w_lat: float = 0.0,
                       w_cost: float = 0.0,
                       w_energy: float = 1.0) -> PlacementSolution:
    """Evaluate a placement and compute all metrics"""
    sol = PlacementSolution(placement=dict(placement))
    
    NG = network_graph.G
    SG = service_graph.G
    nbC = SG.number_of_nodes()
    nbH = NG.number_of_nodes()
    
    # C0: Completeness
    for c in SG.nodes():
        if c not in placement:
            sol.violations.append(f"Component {c} not placed")
            sol.feasible = False
    
    # C1: DZ constraints
    dz_data = service_graph.metadata.get('component.DZ', [])
    if dz_data and len(dz_data) >= 2:
        for i in range(0, len(dz_data), 2):
            if i+1 >= len(dz_data):
                break
            ci, hi = dz_data[i], dz_data[i+1]
            if sol.placement.get(ci) != hi:
                sol.violations.append(f"DZ violation: C{ci} must be on H{hi}")
                sol.feasible = False
    
    # C2: Capacity
    cu = {h: 0 for h in NG.nodes()}
    ru = {h: 0 for h in NG.nodes()}
    
    for c in SG.nodes():
        if c not in sol.placement:
            continue
        h = sol.placement[c]
        cpu_req = SG.nodes[c].get('cpu', 0)
        ram_req = SG.nodes[c].get('ram', 0)
        cu[h] += cpu_req
        ru[h] += ram_req
    
    for h in NG.nodes():
        cpu_cap = NG.nodes[h].get('cpu', 0)
        ram_cap = NG.nodes[h].get('ram', 0)
        if cu[h] > cpu_cap:
            sol.violations.append(f"Host {h} CPU overflow: {cu[h]} > {cpu_cap}")
            sol.feasible = False
        if ru[h] > ram_cap:
            sol.violations.append(f"Host {h} RAM overflow: {ru[h]} > {ram_cap}")
            sol.feasible = False
    
    sol.host_cpu_used = dict(cu)
    sol.host_ram_used = dict(ru)
    sol.active_hosts = {h for h in NG.nodes() if cu[h] > 0}
    
    # C3: Paths + latency per link
    link_lats = []
    arc_flow = defaultdict(int)
    
    for u, v, data in SG.edges(data=True):
        if u not in sol.placement or v not in sol.placement:
            continue
        hs, ht = sol.placement[u], sol.placement[v]
        bw = data.get('bandwidth', 0)
        
        if hs == ht:
            link_lats.append(0)
            sol.paths[len(sol.paths)] = [hs]
            sol.consumed_lat[len(sol.consumed_lat)] = 0
        else:
            try:
                path = nx.shortest_path(NG, source=hs, target=ht, weight='latency')
                path_lat = 0
                for i in range(len(path) - 1):
                    n1, n2 = path[i], path[i+1]
                    lat = NG.edges[n1, n2].get('latency', 0)
                    path_lat += lat
                    arc_flow[(n1, n2)] += bw
                
                link_lats.append(path_lat)
                sol.paths[len(sol.paths)] = path
                sol.consumed_lat[len(sol.consumed_lat)] = path_lat
            except nx.NetworkXNoPath:
                sol.violations.append(f"No path between H{hs} and H{ht} for link C{u}→C{v}")
                sol.feasible = False
                link_lats.append(INF_V)
    
    # C4: Bandwidth capacity
    for (u, v), flow in arc_flow.items():
        if NG.has_edge(u, v):
            bw_cap = NG.edges[u, v].get('bandwidth', 0)
            if bw_cap > 0 and flow > bw_cap:
                sol.violations.append(f"Link H{u}→H{v} bandwidth overflow: {flow} > {bw_cap}")
                sol.feasible = False
    
    # Metrics
    sol.worst_case_latency = max(link_lats) if link_lats else 0
    sol.total_latency = sum(x for x in link_lats if x < INF_V)
    
    # Cost (assuming hosts may have cost attribute, otherwise use CPU as proxy)
    sol.infra_cost = sum(
        NG.nodes[h].get('cost', NG.nodes[h].get('cpu', 1))
        for h in sol.active_hosts
    )
    
    # Energy
    sol.energy_link = sum(
        fl * NG.edges[u, v].get('latency', 0) ** 2
        for (u, v), fl in arc_flow.items()
    )
    sol.energy_node = sum(P_STATIC + cu[h] * P_CPU_UNIT for h in sol.active_hosts)
    sol.energy_total = sol.energy_link + sol.energy_node
    
    # Objective
    if sol.feasible:
        def norm_int(val, ub):
            return int((val * NORM) / max(ub, 1))
        
        lat_n = norm_int(sol.total_latency, bounds.lat_ub)
        cost_n = norm_int(sol.infra_cost, bounds.cost_max)
        ener_n = norm_int(sol.energy_total, bounds.energy_ub)
        
        W = 1000
        wl, wc, we = int(w_lat * W), int(w_cost * W), int(w_energy * W)
        sol.objective_int = lat_n * wl + cost_n * wc + ener_n * we
        sol.objective = float(sol.objective_int)
    else:
        sol.objective_int = 10 ** 9
        sol.objective = float("inf")
    
    return sol


# =====================================================================
# GREEDY BASELINE
# =====================================================================
def _greedy_placement(network_graph: NetworkGraph, 
                      service_graph: ServiceGraph) -> Dict[int, int]:
    """Simple greedy First-Fit Decreasing placement"""
    NG = network_graph.G
    SG = service_graph.G
    
    pl = {}
    cu = {h: 0 for h in NG.nodes()}
    ru = {h: 0 for h in NG.nodes()}
    
    # Handle DZ
    dz_data = service_graph.metadata.get('component.DZ', [])
    if dz_data and len(dz_data) >= 2:
        for i in range(0, len(dz_data), 2):
            if i+1 >= len(dz_data):
                break
            ci, hi = dz_data[i], dz_data[i+1]
            pl[ci] = hi
            cpu_req = SG.nodes[ci].get('cpu', 0)
            ram_req = SG.nodes[ci].get('ram', 0)
            cu[hi] += cpu_req
            ru[hi] += ram_req
    
    # Place remaining components
    for c in SG.nodes():
        if c in pl:
            continue
        
        cpu_req = SG.nodes[c].get('cpu', 0)
        ram_req = SG.nodes[c].get('ram', 0)
        
        # Find first fit
        best_h = None
        for h in NG.nodes():
            cpu_cap = NG.nodes[h].get('cpu', 0)
            ram_cap = NG.nodes[h].get('ram', 0)
            if cu[h] + cpu_req <= cpu_cap and ru[h] + ram_req <= ram_cap:
                best_h = h
                break
        
        if best_h is not None:
            pl[c] = best_h
            cu[best_h] += cpu_req
            ru[best_h] += ram_req
        else:
            # Fallback to first host
            pl[c] = list(NG.nodes())[0]
    
    return pl


# =====================================================================
# SEED PLACEMENT GENERATION
# =====================================================================
def _generate_seed_placements(network_graph: NetworkGraph,
                               service_graph: ServiceGraph,
                               bounds: NormBounds,
                               w: Tuple[float, float, float]) -> List[Tuple[str, Dict[int, int]]]:
    """Generate diverse seed placements to guide LLM"""
    seeds = []
    
    NG = network_graph.G
    SG = service_graph.G
    
    # DZ components
    dz = {}
    dz_data = service_graph.metadata.get('component.DZ', [])
    if dz_data and len(dz_data) >= 2:
        for i in range(0, len(dz_data), 2):
            if i+1 >= len(dz_data):
                break
            dz[dz_data[i]] = dz_data[i+1]
    
    free = [c for c in SG.nodes() if c not in dz]
    
    # Seed 1: Greedy FFD
    seeds.append(("FFD greedy baseline", _greedy_placement(network_graph, service_graph)))
    
    # Seed 2: Consolidate on single best host
    for target_h in NG.nodes():
        pl = dict(dz)
        cu = {h: 0 for h in NG.nodes()}
        ru = {h: 0 for h in NG.nodes()}
        
        # Account for DZ
        for ci, hi in dz.items():
            cpu_req = SG.nodes[ci].get('cpu', 0)
            ram_req = SG.nodes[ci].get('ram', 0)
            cu[hi] += cpu_req
            ru[hi] += ram_req
        
        # Try to place all free on target_h
        free_cpu = sum(SG.nodes[c].get('cpu', 0) for c in free)
        free_ram = sum(SG.nodes[c].get('ram', 0) for c in free)
        
        target_cpu_cap = NG.nodes[target_h].get('cpu', 0)
        target_ram_cap = NG.nodes[target_h].get('ram', 0)
        
        if cu[target_h] + free_cpu <= target_cpu_cap and ru[target_h] + free_ram <= target_ram_cap:
            for c in free:
                pl[c] = target_h
            if len(pl) == len(SG.nodes()):
                seeds.append((f"Consolidate on H{target_h}", pl))
                break
    
    # Seed 3: DZ-spread (place near DZ hosts)
    if dz:
        pl = dict(dz)
        cu = {h: 0 for h in NG.nodes()}
        ru = {h: 0 for h in NG.nodes()}
        
        for ci, hi in dz.items():
            cpu_req = SG.nodes[ci].get('cpu', 0)
            ram_req = SG.nodes[ci].get('ram', 0)
            cu[hi] += cpu_req
            ru[hi] += ram_req
        
        dz_hosts = sorted(set(dz.values()))
        
        for c in free:
            cpu_req = SG.nodes[c].get('cpu', 0)
            ram_req = SG.nodes[c].get('ram', 0)
            
            # Try DZ hosts first
            placed = False
            for dh in dz_hosts:
                cpu_cap = NG.nodes[dh].get('cpu', 0)
                ram_cap = NG.nodes[dh].get('ram', 0)
                if cu[dh] + cpu_req <= cpu_cap and ru[dh] + ram_req <= ram_cap:
                    pl[c] = dh
                    cu[dh] += cpu_req
                    ru[dh] += ram_req
                    placed = True
                    break
            
            if not placed:
                # Try any host
                for h in NG.nodes():
                    cpu_cap = NG.nodes[h].get('cpu', 0)
                    ram_cap = NG.nodes[h].get('ram', 0)
                    if cu[h] + cpu_req <= cpu_cap and ru[h] + ram_req <= ram_cap:
                        pl[c] = h
                        cu[h] += cpu_req
                        ru[h] += ram_req
                        break
        
        if len(pl) == len(SG.nodes()):
            seeds.append(("DZ-spread", pl))
    
    return seeds


# =====================================================================
# OPRO TRAJECTORY & PROMPT BUILDING
# =====================================================================
def _build_opro_trajectory(trajectory: List[Tuple[float, Dict[int, int], str]], 
                           max_shown: int = 20) -> str:
    """Build compact optimization trajectory (worst→best)"""
    if not trajectory:
        return ""
    
    # Keep best N, sort worst→best
    shown = sorted(trajectory, key=lambda x: -x[0])[-max_shown:]
    shown.sort(key=lambda x: x[0], reverse=True)
    
    lines = ["\n=== TRAJECTORY (worst→best, LOWER=BETTER) ==="]
    for i, (score, pl, reasoning) in enumerate(shown):
        active = len(set(pl.values()))
        pl_compact = ",".join(f"{c}:{h}" for c, h in sorted(pl.items()))
        insight = f" | {reasoning[:80]}" if reasoning else ""
        lines.append(f"  #{i+1} score={score} hosts={active} [{pl_compact}]{insight}")
    
    best_score = shown[-1][0] if shown else float("inf")
    lines.append(f"\nBEST={best_score}. Generate NEW placement with score < {best_score}.")
    return "\n".join(lines)


def _build_problem_description(network_graph: NetworkGraph,
                                service_graph: ServiceGraph,
                                bounds: NormBounds,
                                w: Tuple[float, float, float]) -> str:
    """Build structured problem description for LLM"""
    lines = []
    
    NG = network_graph.G
    SG = service_graph.G
    
    # Infrastructure
    lines.append("=== INFRASTRUCTURE ===")
    lines.append(f"Hosts: {NG.number_of_nodes()} | Diameter: {network_graph.metadata.get('network.diameter', 'N/A')}")
    lines.append("\nHost details:")
    for h in NG.nodes():
        cpu = NG.nodes[h].get('cpu', 0)
        ram = NG.nodes[h].get('ram', 0)
        cost = NG.nodes[h].get('cost', cpu)
        tier = "Cloud" if cpu >= 32 else ("Fog" if cpu >= 8 else "Edge")
        lines.append(f"  H{h}: CPU={cpu:>3}, RAM={ram:>6}, Cost={cost:>3}, Tier={tier}")
    
    # Application
    lines.append("\n=== APPLICATION ===")
    lines.append(f"Components: {SG.number_of_nodes()} | Links: {SG.number_of_edges()}")
    lines.append("\nComponent requirements:")
    for c in SG.nodes():
        cpu = SG.nodes[c].get('cpu', 0)
        ram = SG.nodes[c].get('ram', 0)
        lines.append(f"  C{c}: CPU={cpu}, RAM={ram}")
    
    lines.append("\nApplication links:")
    for u, v, data in SG.edges(data=True):
        bw = data.get('bandwidth', 0)
        lines.append(f"  L: C{u} → C{v}, bw={bw}")
    
    # DZ constraints
    dz_data = service_graph.metadata.get('component.DZ', [])
    if dz_data and len(dz_data) >= 2:
        lines.append("\n=== DZ CONSTRAINTS (MANDATORY) ===")
        for i in range(0, len(dz_data), 2):
            if i+1 >= len(dz_data):
                break
            ci, hi = dz_data[i], dz_data[i+1]
            lines.append(f"  C{ci} MUST be on H{hi}")
    
    # Objective weights
    lines.append(f"\n=== OBJECTIVE WEIGHTS ===")
    lines.append(f"w_lat={w[0]}, w_cost={w[1]}, w_energy={w[2]}")
    lines.append(f"Bounds: lat_ub={bounds.lat_ub:.0f}, cost_max={bounds.cost_max:.0f}, energy_ub={bounds.energy_ub:.0f}")
    
    return "\n".join(lines)


def _build_opro_prompt(problem_desc: str, trajectory: List, feedback: str,
                       best: Optional[PlacementSolution], iteration: int,
                       max_iter: int, strategy_hint: str = "") -> str:
    """Build OPRO meta-prompt"""
    parts = [problem_desc]
    
    # Trajectory
    traj_text = _build_opro_trajectory(trajectory)
    if traj_text:
        parts.append(traj_text)
    
    # Best known
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


# =====================================================================
# LLM PLACEMENT CLASS
# =====================================================================
class LLMPlacement(PlacementAlgo):
    """LLM-based placement using OPRO (Optimization by PROmpting)
    Adapeted from the work of Google DeepMind's paper: [LARGE LANGUAGE MODELS AS OPTIMIZERS](https://arxiv.org/pdf/2309.03409) and Farah AIT SALAHT
    """
    def __init__(self, provider: str = "auto", model: str = "auto", 
                 max_iter: int = 5, verbose: bool = True):
        """
        Initialize LLM placement solver.
        
        Args:
            provider: LLM provider ("anthropic", "openai", "gemini", "ollama", "auto")
            model: Model name (or "auto" to use default for provider)
            max_iter: Maximum optimization iterations
            verbose: Print progress
        """
        self.provider = provider
        self.model = model
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Auto-detect provider if needed
        if self.provider == "auto":
            self.provider, self.model = self._auto_detect_provider()
            if self.verbose:
                print(f"  [LLM] Auto-detected: {self.provider} ({self.model})")
    
    def place(self, service_graph: ServiceGraph, network_graph: NetworkGraph, 
              **kwargs) -> PlacementResult:
        """
        Place services using LLM optimization.
        
        Args:
            service_graph: Application service graph
            network_graph: Infrastructure network graph
            **kwargs: Algorithm parameters (w_lat, w_cost, w_energy, etc.)
            
        Returns:
            PlacementResult with optimized placement
        """
        # Extract weights
        w_lat = kwargs.get('w_lat', 0.0)
        w_cost = kwargs.get('w_cost', 0.0)
        w_energy = kwargs.get('w_energy', 1.0)
        w = (w_lat, w_cost, w_energy)
        
        # Compute bounds
        bounds = _compute_norm_bounds(network_graph, service_graph)
        
        # Run optimization
        t0 = time.time()
        best, metadata = self._run_llm_optimization(
            network_graph, service_graph, bounds, w)
        t_total = time.time() - t0
        
        # Build paths for result
        paths = {}
        NG = network_graph.G
        SG = service_graph.G
        
        for u, v in SG.edges():
            if u in best.placement and v in best.placement:
                hs, ht = best.placement[u], best.placement[v]
                if hs != ht:
                    try:
                        path = nx.shortest_path(NG, source=hs, target=ht, weight='latency')
                        paths[(u, v)] = path
                    except nx.NetworkXNoPath:
                        paths[(u, v)] = []
                else:
                    paths[(u, v)] = [hs]
        
        # Add metrics to metadata
        metadata['solve_time'] = t_total
        metadata['objective'] = best.objective_int
        metadata['feasible'] = best.feasible
        metadata['energy_total'] = best.energy_total
        metadata['latency_total'] = best.total_latency
        metadata['cost'] = best.infra_cost
        
        return PlacementResult(
            mapping=best.placement,
            paths=paths,
            meta=metadata
        )
    
    def _run_llm_optimization(self, network_graph: NetworkGraph,
                              service_graph: ServiceGraph,
                              bounds: NormBounds,
                              w: Tuple[float, float, float]) -> Tuple[PlacementSolution, Dict]:
        """Run OPRO-style LLM optimization"""
        total_tok_in = 0
        total_tok_out = 0
        total_llm_time = 0.0
        best = None
        
        # OPRO trajectory
        trajectory = []
        
        # Phase 1: Seed evaluation
        seeds = _generate_seed_placements(network_graph, service_graph, bounds, w)
        if self.verbose:
            print(f"    [LLM] Generated {len(seeds)} seed placements")
        
        for name, pl in seeds:
            sol = _evaluate_solution(pl, network_graph, service_graph, bounds, *w)
            trajectory.append((
                sol.objective_int if sol.feasible else 10**9,
                dict(sol.placement),
                name
            ))
            if sol.feasible and (best is None or sol.objective < best.objective):
                best = sol
                if self.verbose:
                    print(f"      {name}: obj={sol.objective_int} *BEST*")
        
        # Phase 2: Build problem description
        problem_desc = _build_problem_description(network_graph, service_graph, bounds, w)
        
        # Phase 3: OPRO iterations
        opro_strategies = [
            "EXPLOIT: Make ONE small change to improve the best solution.",
            "NEARBY SPREAD: Place components on NEAREST hosts to DZ hosts.",
            "FIX BOTTLENECK: Colocate endpoints of the worst link.",
            "DZ-CENTRIC: Place near highest-bandwidth DZ hosts.",
            "ENERGY MINIMIZER: Focus on minimizing E_links (quadratic in distance).",
            "COST REDUCER: Use cheaper hosts while maintaining performance.",
            "CREATIVE: Try a VERY DIFFERENT placement pattern.",
            "MICRO-OPTIMIZE: Test every single-component move systematically.",
        ]
        
        stale_iterations = 0
        max_stale = 3
        
        for it in range(1, self.max_iter + 1):
            if stale_iterations >= max_stale:
                if self.verbose:
                    print(f"    [LLM] Converged after {it-1} iterations")
                break
            
            # Select strategy
            strat_idx = (it - 1) % len(opro_strategies)
            strat = opro_strategies[strat_idx]
            
            # Temperature schedule
            if it <= self.max_iter // 3:
                temperature = 0.15
            elif it <= 2 * self.max_iter // 3:
                temperature = 0.5
            else:
                temperature = 0.2
            
            # Build prompt
            prompt = _build_opro_prompt(
                problem_desc, trajectory, "", best, it, self.max_iter, strat)
            
            if self.verbose:
                print(f"    [LLM] Iteration {it}/{self.max_iter}: {strat[:40]}...", end=" ")
            
            # Call LLM
            try:
                t_call = time.time()
                text, ti, to = _call_llm_with_tokens(
                    prompt, LLM_SYSTEM, self.provider, self.model, temperature)
                dt = time.time() - t_call
                total_tok_in += ti
                total_tok_out += to
                total_llm_time += dt
            except Exception as e:
                if self.verbose:
                    print(f"ERROR: {e}")
                stale_iterations += 1
                continue
            
            # Parse response
            pl = _parse_llm_json(text, service_graph.G.number_of_nodes())
            if pl is None:
                if self.verbose:
                    print("PARSE FAIL")
                stale_iterations += 1
                continue
            
            # Evaluate
            sol = _evaluate_solution(pl, network_graph, service_graph, bounds, *w)
            
            # Extract reasoning
            reasoning = ""
            try:
                m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
                if m:
                    reasoning = m.group(1)[:120]
            except Exception:
                pass
            
            # Add to trajectory
            trajectory.append((
                sol.objective_int if sol.feasible else 10**9,
                dict(sol.placement),
                reasoning or f"LLM iter {it}"
            ))
            
            # Check improvement
            step_improved = False
            if sol.feasible and (best is None or sol.objective < best.objective):
                best = sol
                step_improved = True
                stale_iterations = 0
                if self.verbose:
                    print(f"obj={sol.objective_int} *BEST*")
            else:
                stale_iterations += 1
                if self.verbose:
                    print(f"obj={sol.objective_int}")
        
        # Fallback to greedy if no solution
        if best is None:
            greedy_pl = _greedy_placement(network_graph, service_graph)
            best = _evaluate_solution(greedy_pl, network_graph, service_graph, bounds, *w)
        
        # Compute energy and cost
        total_tok = total_tok_in + total_tok_out
        e_per_1k = ENERGY_PER_1K_TOKENS.get(self.provider, 0.004)
        cost_in, cost_out = COST_PER_1K_TOKENS.get(self.provider, (0, 0))
        
        if self.provider == "ollama":
            e_solver = P_CPU_LOCAL * total_llm_time / 3600.0
        else:
            e_solver = total_tok / 1000.0 * e_per_1k
        
        cost_usd = total_tok_in / 1000.0 * cost_in + total_tok_out / 1000.0 * cost_out
        
        if self.verbose:
            print(f"  [LLM] Tokens: {total_tok_in}+{total_tok_out}={total_tok}")
            print(f"  [LLM] E_solver={e_solver:.6f}Wh | Cost=${cost_usd:.4f}")
        
        metadata = {
            'tokens_in': total_tok_in,
            'tokens_out': total_tok_out,
            'total_tokens': total_tok,
            'e_solver_wh': e_solver,
            'cost_usd': cost_usd,
            'llm_time': total_llm_time,
            'provider': self.provider,
            'model': self.model,
        }
        
        return best, metadata
    
    def _auto_detect_provider(self) -> Tuple[str, str]:
        """Auto-detect available LLM provider"""
        # Try Gemini
        gemini_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if gemini_key:
            return "gemini", os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
        
        # Try Anthropic
        if os.environ.get("ANTHROPIC_API_KEY", ""):
            return "anthropic", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        
        # Try OpenAI
        if os.environ.get("OPENAI_API_KEY", ""):
            return "openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        
        # Try Ollama
        models = self._ollama_list_models()
        if models:
            for pref in ["gemma2", "llama3", "mistral", "qwen2"]:
                for m in models:
                    if pref in m:
                        return "ollama", m
            return "ollama", models[0]

        raise ValueError(
            "No LLM provider detected. Set API keys in environment variables using:\n"
            "- For Gemini: GEMINI_API_KEY or GOOGLE_API_KEY\n"
            "- For Anthropic: ANTHROPIC_API_KEY\n"
            "- For OpenAI: OPENAI_API_KEY\n"
            "or install Ollama.\n"
            "On Windows, use 'set VAR=value' and on Linux/Mac use 'export VAR=value' in the terminal."
        )
    
    def _ollama_list_models(self) -> List[str]:
        """List available Ollama models"""
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
                        return models
            except Exception:
                pass
        return []
