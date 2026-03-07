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
from functools import lru_cache

import networkx as nx

from src.placementAlgo import PlacementAlgo, PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph

# =====================================================================
# CONSTANTS
# =====================================================================
NORM = 10000
INF_V = 10 ** 9
P_CPU_LOCAL = float(os.environ.get("P_CPU_WATTS", "45"))  # Watts

# GCP energy model defaults (aligned with Code_unified_benchmark_energy_gcp.py)
GCP_PUE = 1.10
GCP_P_VCPU_IDLE_W = 1.0
GCP_P_VCPU_ACTIVE_W = 8.0

GCP_LAT_INTRAZONE_MS = 2.0
GCP_LAT_INTERZONE_MS = 25.0

GCP_FACTOR_INTRAZONE = 1.0
GCP_FACTOR_INTERZONE = 1.5
GCP_FACTOR_CROSSREGION = 2.0


def _parse_simple_properties(file_path: str) -> Dict[str, str]:
    """Minimal .properties parser (key=value), compatible with project files."""
    props: Dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        continuation = ""
        for raw in f:
            line = raw.rstrip("\n")
            if line.endswith("\\"):
                continuation += line[:-1]
                continue
            line = (continuation + line).strip()
            continuation = ""
            if not line or line[0] in "#!":
                continue
            if "=" in line:
                k, v = line.split("=", 1)
            elif ":" in line:
                k, v = line.split(":", 1)
            else:
                continue
            props[k.strip()] = v.strip()
    return props


@lru_cache(maxsize=1)
def _load_energy_settings() -> Dict[str, float]:
    """Load GCP energy constants from Energy_GCP.properties with safe defaults."""
    default_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "properties", "Energy_GCP.properties")
    )
    config_path = os.environ.get("CSP_ENERGY_PROPERTIES", "") or default_path

    settings = {
        "gcp.pue": GCP_PUE,
        "gcp.p_vcpu_idle_w": GCP_P_VCPU_IDLE_W,
        "gcp.p_vcpu_active_w": GCP_P_VCPU_ACTIVE_W,
        "gcp.lat.intrazone_ms": GCP_LAT_INTRAZONE_MS,
        "gcp.lat.interzone_ms": GCP_LAT_INTERZONE_MS,
        "gcp.factor.intrazone": GCP_FACTOR_INTRAZONE,
        "gcp.factor.interzone": GCP_FACTOR_INTERZONE,
        "gcp.factor.crossregion": GCP_FACTOR_CROSSREGION,
    }

    if os.path.exists(config_path):
        try:
            raw = _parse_simple_properties(config_path)
            for key in list(settings.keys()):
                if key in raw:
                    settings[key] = float(raw[key])
        except Exception:
            pass

    return settings


def _link_factor(latency_ms: float, cfg: Dict[str, float]) -> float:
    """Infer GCP link factor from measured latency tier."""
    if latency_ms <= cfg["gcp.lat.intrazone_ms"]:
        return cfg["gcp.factor.intrazone"]
    if latency_ms <= cfg["gcp.lat.interzone_ms"]:
        return cfg["gcp.factor.interzone"]
    return cfg["gcp.factor.crossregion"]


def _node_power_w(cpu_used: int, cpu_cap: int, cfg: Dict[str, float]) -> float:
    """GCP vCPU-slot node model with explicit zero for inactive hosts."""
    if cpu_cap <= 0 or cpu_used <= 0:
        return 0.0
    used = min(cpu_used, cpu_cap)
    return cfg["gcp.pue"] * (
        cfg["gcp.p_vcpu_idle_w"] * cpu_cap
        + (cfg["gcp.p_vcpu_active_w"] - cfg["gcp.p_vcpu_idle_w"]) * used
    )

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
LLM_SYSTEM = """You are an expert optimizer for the Service Placement Problem (SPP) on Google Cloud Platform (GCP).
The infrastructure consists exclusively of GCE VMs (e2/n2/c2 families). There are NO fog nodes, NO edge devices.
Your goal is to find a placement that MINIMIZES the ENERGY objective. Lower is better.

OBJECTIVE FUNCTION (ENERGY ONLY):
  obj = P_total = P_links + P_nodes  (continuous power [W], datastream regime)

WHERE:
- P_links = SUM over all application links l (GCP-aware model):
    bandwidth[l] × sp_latency(hs_l, ht_l) × factor(link_type)
  Link type inferred from measured latency:
    intra-zone  (sp_latency ≤ 2ms)  : factor = 1.0  — same datacenter fabric
    inter-zone  (sp_latency ≤ 25ms) : factor = 1.5  — regional backbone
    cross-region (sp_latency > 25ms) : factor = 2.0  — WAN long-haul (VERY EXPENSIVE)
  If both endpoints are on the SAME VM, E_link for that link = 0.

- P_nodes = SUM over all ACTIVE VMs h (GCP vCPU-slot model, SPECpower calibration):
    GCP_PUE × [P_vcpu_idle × cpu_cap_h + (P_vcpu_active - P_vcpu_idle) × cpu_used_h]
  GCP_PUE = 1.10 (uniform, Google Environmental Report 2023)
  P_vcpu_idle = 1.0 W/vCPU,  P_vcpu_active = 8.0 W/vCPU
  Examples:
    e2-standard-2  (2 vCPU) idle=2.2W,  full=17.6W
    n2-standard-8  (8 vCPU) idle=8.8W,  full=70.4W
    n2-standard-32 (32vCPU) idle=35.2W, full=281.6W
  Inactive VMs (cpu_used=0) consume P=0 (VM stopped).

GCP CRITICAL OPTIMIZATION PRINCIPLES (ranked by impact on ENERGY):
1. SAME-ZONE COLOCATION = ZERO LINK ENERGY: Components on the SAME VM → E_link=0.
   Always prefer colocation when capacity (CPU+RAM) allows.
2. CROSS-REGION IS VERY EXPENSIVE: factor=2.0 × high latency = dominant cost.
   A 100Mbps link across 85ms cross-region costs 100×85×2.0=17000 vs 100×1×1.0=100 intra-zone.
   ALWAYS prioritize moving cross-region links to same zone.
3. STAY IN SAME ZONE: Prefer VMs with sp_latency ≤ 2ms from DZ host (factor=1.0).
   Inter-zone (≤25ms, factor=1.5) is acceptable. Cross-region (>25ms) must be avoided.
4. GCP-SMALL VMs ARE EFFICIENT: e2 VMs (cpu<8) idle at 1.1W/vCPU.
   Prefer small VMs over large when communication is local (intra-zone).
5. FEWER ACTIVE VMs = LOWER E_nodes: Consolidate when all VMs are in same zone.
6. DZ CONSTRAINTS ARE ABSOLUTE: Fixed components MUST stay on their assigned VM.
7. CAPACITY IS A HARD WALL: CPU sum <= cpuCap[h] AND RAM sum <= ramCap[h] per VM.

GCP PLACEMENT STRATEGY:
1. First: can ALL free components fit on the DZ VM (H0) or an intra-zone VM? → best option.
2. Second: if not, use 2 VMs both in the same zone (sp_latency ≤ 2ms from H0).
3. Third: if not, use VMs in the same region (sp_latency ≤ 25ms). 
4. NEVER: place components on cross-region VMs (sp_latency > 25ms) unless forced.

OPTIMIZATION TRAJECTORY:
Below you will see PREVIOUS placement attempts with their objective scores (ENERGY in Watts), sorted from worst to best.
Study the patterns: solutions with LOWER scores use same-zone VMs and avoid cross-region links.
Generate a NEW placement that achieves a LOWER objective than ALL previous attempts.

REASONING PROTOCOL:
1. Examine the FEEDBACK — which links are cross-region (factor=2.0)?
2. For each cross-region link: can both endpoints move to the same zone?
3. VERIFY: CPU+RAM capacity on the target zone VM.
4. Compute expected savings: bw × lat × (2.0 - 1.0) per cross-region link eliminated.
5. Output the improved placement.

HARD CONSTRAINTS (any violation = infeasible = rejected):
- DZ: component must be on its assigned VM
- CPU: sum of cpuComp on VM h <= cpuCap[h]
- RAM: sum of ramComp on VM h <= ramCap[h]
- Connectivity: path must exist between VMs of linked components

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
    cfg = _load_energy_settings()
    
    NG = network_graph.G
    SG = service_graph.G
    
    # Latency upper bound (sum of per-link SP latency worst case)
    lengths = dict(nx.all_pairs_dijkstra_path_length(NG, weight='latency'))
    lat_diameter = max(
        (lengths[u][v] for u in lengths for v in lengths[u]),
        default=float(network_graph.metadata.get('network.diameter', len(NG.nodes())))
    )
    nbL = SG.number_of_edges()
    bounds.lat_ub = max(1, nbL * lat_diameter)
    
    # Cost upper bound (assuming all hosts may have a cost attribute, default to CPU)
    total_cost = sum(
        NG.nodes[h].get('cost', NG.nodes[h].get('cpu', 1)) 
        for h in NG.nodes()
    )
    bounds.cost_max = max(1, total_cost)
    
    # Energy bounds (GCP model)
    max_flow = sum(d.get('bandwidth', 0) for _, _, d in SG.edges(data=True))  
    max_flow = max(1, max_flow)
    bounds.e_link_ub = max(1, max_flow * lat_diameter * cfg["gcp.factor.crossregion"])

    max_p_eff = max(
        (cfg["gcp.pue"] * cfg["gcp.p_vcpu_active_w"] * (NG.nodes[h].get('cpu', 0) or 0)
         for h in NG.nodes()),
        default=1.0
    )
    bounds.e_node_ub = max(1, int(len(NG.nodes()) * max_p_eff))
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
    cfg = _load_energy_settings()
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
        if h not in cu:
            sol.violations.append(f"Invalid host assignment: C{c} -> H{h} (host does not exist)")
            sol.feasible = False
            continue
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
        lat_ub = data.get('latency', float('inf'))
        
        if hs == ht:
            # Colocated case: path_lat = 0, which should always satisfy latency constraint
            path_lat = 0
            # Enforce latency constraint explicitly
            if path_lat > lat_ub:
                sol.violations.append(
                    f"Latency violation on link C{u}->C{v}: {path_lat} > {lat_ub}"
                )
                sol.feasible = False

            link_lats.append(path_lat)
            sol.paths[len(sol.paths)] = [hs]
            sol.consumed_lat[len(sol.consumed_lat)] = path_lat
        else:
            try:
                path = nx.shortest_path(NG, source=hs, target=ht, weight='latency')
                path_lat = 0
                for i in range(len(path) - 1):
                    n1, n2 = path[i], path[i+1]
                    lat = NG.edges[n1, n2].get('latency', 0)
                    path_lat += lat
                    arc_flow[(n1, n2)] += bw

                # Enforce latency constraint: path latency must not exceed limit
                if path_lat > lat_ub:
                    sol.violations.append(
                        f"Latency violation on link C{u}→C{v}: {path_lat} > {lat_ub}"
                    )
                    sol.feasible = False
                
                link_lats.append(path_lat)
                sol.paths[len(sol.paths)] = path
                sol.consumed_lat[len(sol.consumed_lat)] = path_lat
            except nx.NetworkXNoPath:
                sol.violations.append(f"No path between H{hs} and H{ht} for link C{u}->C{v}")
                sol.feasible = False
                link_lats.append(INF_V)
    
    # C4: Bandwidth capacity
    for (u, v), flow in arc_flow.items():
        if NG.has_edge(u, v):
            bw_cap = NG.edges[u, v].get('bandwidth', 0)
            if bw_cap > 0 and flow > bw_cap:
                sol.violations.append(f"Link H{u}->H{v} bandwidth overflow: {flow} > {bw_cap}")
                sol.feasible = False
    
    # Metrics
    sol.worst_case_latency = max(link_lats) if link_lats else 0
    sol.total_latency = sum(x for x in link_lats if x < INF_V)
    
    # Cost (assuming hosts may have cost attribute, otherwise use CPU as proxy)
    sol.infra_cost = sum(
        NG.nodes[h].get('cost', NG.nodes[h].get('cpu', 1))
        for h in sol.active_hosts
    )
    
    # Energy (GCP model)
    sol.energy_link = sum(
        fl * NG.edges[u, v].get('latency', 0) * _link_factor(float(NG.edges[u, v].get('latency', 0)), cfg)
        for (u, v), fl in arc_flow.items()
    )
    sol.energy_node = sum(
        _node_power_w(cu[h], int(NG.nodes[h].get('cpu', 0) or 0), cfg)
        for h in sol.active_hosts
    )
    sol.energy_total = sol.energy_link + sol.energy_node
    
    # Objective (ENERGY ONLY - direct raw value in Watts)
    if sol.feasible:
        # Raw energy in Watts (P_total = P_links + P_nodes)
        sol.objective = sol.energy_total
        # Convert to milliwatts for integer comparison (preserve precision)
        sol.objective_int = int(sol.energy_total * 1000)
    else:
        sol.objective_int = 10 ** 9
        sol.objective = float("inf")
    
    return sol


# =====================================================================
# GREEDY BASELINE
# =====================================================================
def _greedy_placement(network_graph: NetworkGraph, 
                      service_graph: ServiceGraph) -> Dict[int, int]:
    """Latency-aware greedy placement used as robust fallback for LLM."""
    NG = network_graph.G
    SG = service_graph.G
    cfg = _load_energy_settings()

    # Precompute shortest-path latencies once to avoid repeated graph traversals.
    sp_len = dict(nx.all_pairs_dijkstra_path_length(NG, weight='latency'))
    
    pl = {}
    cu = {h: 0 for h in NG.nodes()}
    ru = {h: 0 for h in NG.nodes()}
    
    # Handle DZ
    dz_data = service_graph.metadata.get('component.DZ', [])
    dz_hosts = set()
    if dz_data and len(dz_data) >= 2:
        for i in range(0, len(dz_data), 2):
            if i+1 >= len(dz_data):
                break
            ci, hi = dz_data[i], dz_data[i+1]
            pl[ci] = hi
            dz_hosts.add(hi)
            cpu_req = SG.nodes[ci].get('cpu', 0)
            ram_req = SG.nodes[ci].get('ram', 0)
            cu[hi] += cpu_req
            ru[hi] += ram_req

    # Place remaining components in decreasing CPU order.
    comp_order = sorted(
        [c for c in SG.nodes() if c not in pl],
        key=lambda c: (SG.nodes[c].get('cpu', 0), SG.degree(c)),
        reverse=True,
    )

    for c in comp_order:
        if c in pl:
            continue
        
        cpu_req = SG.nodes[c].get('cpu', 0)
        ram_req = SG.nodes[c].get('ram', 0)
        
        best_h = None
        best_score = float("inf")

        for h in NG.nodes():
            cpu_cap = NG.nodes[h].get('cpu', 0)
            ram_cap = NG.nodes[h].get('ram', 0)
            if cu[h] + cpu_req <= cpu_cap and ru[h] + ram_req <= ram_cap:
                # Node power after placing c on h
                projected_cpu = cu[h] + cpu_req
                node_cost = _node_power_w(projected_cpu, int(cpu_cap or 0), cfg)

                # Communication cost to already placed neighbors (bw * lat * factor)
                comm_cost = 0.0
                for _, nb, data in SG.out_edges(c, data=True):
                    if nb in pl:
                        nb_h = pl[nb]
                        lat = sp_len.get(h, {}).get(nb_h, INF_V)
                        if lat >= INF_V:
                            comm_cost += 1e9
                        else:
                            bw = data.get('bandwidth', 0)
                            comm_cost += bw * lat * _link_factor(float(lat), cfg)
                for nb, _, data in SG.in_edges(c, data=True):
                    if nb in pl:
                        nb_h = pl[nb]
                        lat = sp_len.get(nb_h, {}).get(h, INF_V)
                        if lat >= INF_V:
                            comm_cost += 1e9
                        else:
                            bw = data.get('bandwidth', 0)
                            comm_cost += bw * lat * _link_factor(float(lat), cfg)

                # Slight preference to stay near DZ anchors to avoid pathological H0 consolidation.
                dz_cost = 0.0
                if dz_hosts:
                    vals = [sp_len.get(h, {}).get(dh, INF_V) for dh in dz_hosts]
                    if any(v >= INF_V for v in vals):
                        dz_cost = 1e6
                    else:
                        dz_cost = sum(vals) / max(1, len(vals))

                score = comm_cost + 0.2 * node_cost + 5.0 * dz_cost
                if score < best_score:
                    best_score = score
                    best_h = h
        
        if best_h is not None:
            pl[c] = best_h
            cu[best_h] += cpu_req
            ru[best_h] += ram_req
        else:
            # Fallback to first host
            fallback_h = list(NG.nodes())[0]
            pl[c] = fallback_h
            cu[fallback_h] += cpu_req
            ru[fallback_h] += ram_req
    
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
    
    # Seed 2: Consolidate on selected hosts (do not stop at first feasible host)
    consolidate_candidates = []
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
                sol = _evaluate_solution(pl, network_graph, service_graph, bounds, *w)
                consolidate_candidates.append((sol.objective_int if sol.feasible else 10**9, target_h, pl))

    consolidate_candidates.sort(key=lambda x: x[0])
    for _, target_h, pl in consolidate_candidates[:3]:
        seeds.append((f"Consolidate on H{target_h}", pl))
    
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
        return "\n=== NO TRAJECTORY YET ==="
    
    # Keep best N, sort worst→best
    shown = sorted(trajectory, key=lambda x: -x[0])[-max_shown:]
    shown.sort(key=lambda x: x[0], reverse=True)
    
    lines = ["\n=== TRAJECTORY (worst→best, LOWER ENERGY=BETTER) ==="]
    for i, (score, pl, reasoning) in enumerate(shown):
        active = len(set(pl.values()))
        pl_compact = ",".join(f"{c}:{h}" for c, h in sorted(pl.items()))
        insight = f" | {reasoning[:80]}" if reasoning else ""
        lines.append(f"  #{i+1} Energy={score:.2f}W hosts={active} [{pl_compact}]{insight}")
    
    best_score = shown[-1][0] if shown else float("inf")
    lines.append(f"\nBEST ENERGY={best_score:.2f}W. Generate NEW placement with LOWER energy.")
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
        tier = "gcp-large" if cpu >= 32 else ("gcp-medium" if cpu >= 8 else "gcp-small")
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
    
    # Objective (ENERGY ONLY)
    lines.append(f"\n=== OBJECTIVE: MINIMIZE ENERGY (Watts) ===")
    lines.append(f"obj = P_total = P_links + P_nodes")
    cfg = _load_energy_settings()
    lines.append(f"\nEnergy model GCP:")
    lines.append(
        f"  P_links: bw × sp_latency × factor  (factor: intra-zone={cfg['gcp.factor.intrazone']:.1f}, "
        f"inter-zone={cfg['gcp.factor.interzone']:.1f}, cross-region={cfg['gcp.factor.crossregion']:.1f})"
    )
    lines.append(
        f"  P_nodes: GCP_PUE × [P_vcpu_idle×cpu_cap + (P_vcpu_active-P_vcpu_idle)×cpu_used]  "
        f"(PUE={cfg['gcp.pue']:.2f}, P_idle={cfg['gcp.p_vcpu_idle_w']:.1f}W/vCPU, P_active={cfg['gcp.p_vcpu_active_w']:.1f}W/vCPU)"
    )
    
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
        parts.append(f"\n--- BEAT THIS: Energy={best.objective:.2f}W "
                     f"(P_links={best.energy_link:.2f}W + P_nodes={best.energy_node:.2f}W) "
                     f"hosts={sorted(best.active_hosts)} ---")
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
    Adapted from the work of Google DeepMind's paper: [LARGE LANGUAGE MODELS AS OPTIMIZERS](https://arxiv.org/pdf/2309.03409) and Farah AIT SALAHT
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
        metadata['status'] = 'success' if best.feasible else 'failed'
        if not best.feasible:
            metadata['violations'] = list(best.violations)
            metadata['reason'] = (
                "; ".join(best.violations[:5])
                if best.violations
                else "No feasible placement found"
            )
        
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
                sol.objective if sol.feasible else float('inf'),
                dict(sol.placement),
                name
            ))
            if sol.feasible and (best is None or sol.objective < best.objective):
                best = sol
                if self.verbose:
                    print(f"      {name}: Energy={sol.objective:.2f}W *BEST*")
        
        # Phase 2: Build problem description
        problem_desc = _build_problem_description(network_graph, service_graph, bounds, w)
        
        # Phase 3: OPRO iterations
        opro_strategies = [
            "EXPLOIT: Make ONE small change to improve the best solution.",
            "NEARBY SPREAD: Place components on NEAREST hosts to DZ hosts.",
            "FIX BOTTLENECK: Colocate endpoints of the worst link.",
            "DZ-CENTRIC: Place near highest-bandwidth DZ hosts.",
            "ENERGY MINIMIZER: Reduce cross-region links first, then inter-zone, then consolidate.",
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
            
            # Add to trajectory (use raw energy for sorting/display)
            trajectory.append((
                sol.objective if sol.feasible else float('inf'),
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
                    print(f"Energy={sol.objective:.2f}W *BEST*")
            else:
                stale_iterations += 1
                if self.verbose:
                    print(f"Energy={sol.objective:.2f}W")
        
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
