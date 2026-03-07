#!/usr/bin/env python3
"""
Code Placement - Benchmark Unifie : CSP / LLM (OPRO) / Glouton
GCP Cloud-Pure Edition (v10-gcp)
========================================================================
Objectif principal: ENERGIE (E_nodes + E_links)
  - Greedy: min ΔE_total marginal (FFD tri CPU décroissant)
  - CSP: min E_total normalisé (CP-SAT, w_energy=1)
  - LLM: OPRO itératif (system prompt énergie GCP-aware)

CONTEXTE: Infrastructure Google Cloud Platform (GCP) — VMs GCE uniquement.
  Aucun nœud fog/edge. Hiérarchie réseau : intra-zone < inter-zone < cross-région.

========================================================================

MODÈLE ÉNERGÉTIQUE — Références bibliographiques
========================================================================
[1] Barroso, L.A., Clidaras, J., Hölzle, U. (2013).
    The Datacenter as a Computer, 2nd ed. Morgan & Claypool.
    → Modèle linéaire P_idle + (P_max - P_idle) * u

[2] Google LLC (2023). Google Environmental Report 2023.
    https://sustainability.google/reports/google-2023-environmental-report/
    → PUE moyen GCP = 1.10 (mesuré sur tous datacenters Google)

[3] Kliazovich, D., Bouvry, P., Khan, S.U. (2012).
    GreenCloud: A Packet-Level Simulator of Energy-Aware Cloud Computing
    Data Centers. J. Supercomputing, 62, 1263-1283.
    → Modèle réseau basé trafic/équipements actifs

[4] SPECpower_ssj2008 (2023). Standard Performance Evaluation Corporation.
    https://www.spec.org/power_ssj2008/
    → Valeurs P_idle/P_max calibrées par vCPU-slot sur serveurs GCE

[5] Ait Salaht, F., Desprez, F., Lebre, A., Prud'Homme, C., Abderrahim, M.
    (2019). Service Placement in Fog Computing Using Constraint Programming.
    IEEE SCC, 19-27.
    → Modèle CSP de référence, E_link linéaire bw*lat

[6] Forti, S., Brogi, A. (2021).
    Continuous Reasoning for Managing Next-Gen Distributed Applications.
    ICLP Technical Communications.
    → Modèle : E_link = bw * latency (étendu avec facteurs hiérarchiques GCP)

[7] Agarwal, S. (2018). Public Cloud Inter-region Network Latency as Heat-maps.
    Medium / BigBitBus.
    → Latences mesurées GCP intra-zone (1ms), inter-zone (8ms), cross-région (85ms)

[8] Luccioni, A.S., Viguier, S., Ligozat, A.-L. (2023).
    Estimating the Carbon Footprint of BLOOM. JMLR, 24(253), 1-15.
    → Énergie d'inférence LLM par token

[9] Patterson, D. et al. (2022).
    The Carbon Footprint of Machine Learning Training Will Plateau, Then Shrink.
    IEEE Micro, 42(4), 18-28.
    → Fourchettes d'énergie inférence LLM, incertitude facteur 3-5x

CHOIX DE MODÉLISATION — GCP Cloud Pur
========================================================================
E_NODE : Modèle vCPU-slot calibré SPECpower [4], PUE GCP uniforme [2].
  P(h) = GCP_PUE × [P_vcpu_idle × cpu_cap + (P_vcpu_active - P_vcpu_idle) × cpu_used]
  PUE uniforme : GCP_PUE = 1.10  (Google Environmental Report 2023 [2])
  Calibration SPECpower_ssj2008 [4] :
    P_vcpu_idle   = 1.0 W/vCPU   (idle baseline)
    P_vcpu_active = 8.0 W/vCPU   (pleine charge)
  Tiers GCP (pour affichage uniquement, PUE identique) :
    gcp-large  (cpu >= 32) : VMs n2/c2 haute densité
    gcp-medium (8 <= cpu < 32) : VMs n2-standard classiques
    gcp-small  (cpu < 8)  : VMs e2 économiques

E_LINK : Proxy hiérarchique bw × latency × factor(type) [5][6][7].
  Facteurs par type de lien (inférés depuis latence mesurée) :
    intra-zone  (lat ≤ 2ms)  : factor = 1.0  (réseau local datacenter)
    inter-zone  (lat ≤ 25ms) : factor = 1.5  (backbone intra-région)
    cross-région (lat > 25ms) : factor = 2.0  (WAN long-haul, très coûteux)
  Colocalisation → E_link = 0 (toujours optimal)

E_SOLVER LLM : tokens_in × eps_in + tokens_out × eps_out [8][9].
  Incertitude ×3 reportée (variabilité datacenter) [9].

ROI/PAYBACK : régime datastream (application permanente).
  T* [s] = E_solver[Wh] / ΔP[W] × 3600
========================================================================
"""

import re, json, time, random, heapq, os, sys, argparse
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field

# =====================================================================
# CONSTANTES GLOBALES
# =====================================================================
NORM = 10000
INF_V = 10 ** 9

# Puissance CPU du solveur local (CSP, Greedy) — configurable via env
# Valeur typique : laptop 15-45W, serveur 45-150W
P_CPU_SOLVER_W = float(os.environ.get("P_CPU_WATTS", "45.0"))

# =====================================================================
# MODÈLE ÉNERGÉTIQUE GCP — Cloud Pur
# Réf: Google Environmental Report 2023 [2], SPECpower_ssj2008 [4]
#
# Formule par nœud (modèle vCPU-slot) :
#   P(h) = GCP_PUE × [P_vcpu_idle × cpu_cap + (P_vcpu_active - P_vcpu_idle) × cpu_used]
#
# PUE uniforme GCP = 1.10 [2] (toutes régions, mesuré Google 2023)
# Calibration SPECpower_ssj2008 [4] : P_vcpu_idle=1.0W, P_vcpu_active=8.0W
#
# Tiers GCP (labeling uniquement, PUE identique) :
#   gcp-large  (cpu >= 32) : n2/c2 haute densité
#   gcp-medium (8<=cpu<32) : n2-standard classiques
#   gcp-small  (cpu < 8)   : e2 économiques
# =====================================================================

# PUE uniforme Google Cloud [2]
GCP_PUE = 1.10

# Puissance par vCPU-slot — calibration SPECpower_ssj2008 [4]
GCP_P_VCPU_IDLE_W   = 1.0   # W/vCPU au repos
GCP_P_VCPU_ACTIVE_W = 8.0   # W/vCPU à pleine charge

# Seuils latence réseau GCP pour inférence du type de lien [7]
GCP_LAT_INTRAZONE_MS  = 2    # ≤ 2ms  → intra-zone
GCP_LAT_INTERZONE_MS  = 25   # ≤ 25ms → inter-zone
# > 25ms → cross-région

# Facteurs énergétiques réseau GCP [3][7]
GCP_FACTOR_INTRAZONE   = 1.0   # réseau local datacenter
GCP_FACTOR_INTERZONE   = 1.5   # backbone intra-région
GCP_FACTOR_CROSSREGION = 2.0   # WAN long-haul (très coûteux)


def get_gcp_tier_label(cpu_cap: int) -> str:
    """Retourne le label de tier GCP pour affichage."""
    if cpu_cap >= 32:
        return "gcp-large"
    elif cpu_cap >= 8:
        return "gcp-medium"
    else:
        return "gcp-small"


def get_tier_params(cpu_cap: int):
    """
    Compatibilité API : retourne (p_idle_W, p_max_W, pue, label) pour un hôte GCP.
    Formule vCPU-slot [4] : p_idle = P_vcpu_idle * cpu_cap,
                             p_max  = P_vcpu_active * cpu_cap.
    PUE uniforme GCP_PUE [2].
    """
    p_idle = GCP_P_VCPU_IDLE_W * cpu_cap
    p_max  = GCP_P_VCPU_ACTIVE_W * cpu_cap
    label  = get_gcp_tier_label(cpu_cap)
    return p_idle, p_max, GCP_PUE, label


def node_power_w(cpu_used: int, cpu_cap: int) -> float:
    """
    Puissance d'une VM GCE en Watts — modèle vCPU-slot [4] :
      P(h) = GCP_PUE × [P_vcpu_idle × cpu_cap + (P_vcpu_active - P_vcpu_idle) × cpu_used]

    Justification : chaque vCPU consomme P_idle au repos même si non utilisé
    (overhead hyperviseur, NUMA, clock gating partiel). [4][2]

    Les VMs éteintes (cpu_used=0) sont supposées arrêtées → P=0.
    """
    if cpu_cap <= 0 or cpu_used <= 0:
        return 0.0
    cpu_used_clamped = min(cpu_used, cpu_cap)
    p = GCP_P_VCPU_IDLE_W * cpu_cap + (GCP_P_VCPU_ACTIVE_W - GCP_P_VCPU_IDLE_W) * cpu_used_clamped
    return GCP_PUE * p


def infer_link_type(latency_ms: int) -> tuple:
    """
    Infère le type de lien GCP depuis la latence mesurée [7].
    Retourne (type_str, factor).
    """
    if latency_ms <= GCP_LAT_INTRAZONE_MS:
        return "intra-zone", GCP_FACTOR_INTRAZONE
    elif latency_ms <= GCP_LAT_INTERZONE_MS:
        return "inter-zone", GCP_FACTOR_INTERZONE
    else:
        return "cross-region", GCP_FACTOR_CROSSREGION


def link_energy_proxy(bw: int, latency_ms: int) -> float:
    """
    Modèle d'énergie réseau GCP-aware — formulation hiérarchique [5][6][7] :
      E_link(l) = bw_l × latency_l × factor(type)

    Facteurs par type (inférés depuis latence mesurée [7]) :
      intra-zone  (≤ 2ms)  : factor=1.0
      inter-zone  (≤ 25ms) : factor=1.5
      cross-région (>25ms) : factor=2.0

    Justification [7] : cross-région implique WAN long-haul (optique transocéanique),
    équipements intermédiaires supplémentaires → facteur énergétique 2× vs intra-zone.

    Retourne une valeur normalisée (float).
    """
    if latency_ms <= 0:
        return 0.0
    _, factor = infer_link_type(latency_ms)
    return float(bw) * float(latency_ms) * factor

# =====================================================================
# MODÈLE ÉNERGÉTIQUE LLM — Réf: Luccioni et al. [8], Patterson et al. [9]
#
# Valeurs en Wh / 1k tokens (estimations médianes) :
#   - Input  : moins coûteux (lecture KV-cache)
#   - Output : ~3x plus coûteux (génération auto-régressive) [9]
# Incertitude : facteur ×3 (variabilité datacenter/GPU/batch) [9]
# =====================================================================

# (energy_input_wh_per_1k, energy_output_wh_per_1k, uncertainty_factor)
LLM_ENERGY_PROFILES = {
    # Anthropic Claude Sonnet — estimation modèle ~70B [8][9]
    "anthropic": (0.004, 0.012, 3.0),
    # OpenAI GPT-4o — modèle large >100B [9]
    "openai":    (0.006, 0.018, 3.0),
    # Google Gemini Flash — modèle léger [9]
    "gemini":    (0.002, 0.006, 3.0),
    # Ollama local — mesuré par P*T si possible, sinon fallback tokens
    "ollama":    (0.003, 0.009, 1.5),
}

# Coûts USD / 1k tokens (input, output)
COST_PER_1K_TOKENS = {
    "anthropic": (0.003, 0.015),
    "openai":    (0.005, 0.015),
    "gemini":    (0.0005, 0.0015),
    "ollama":    (0.0, 0.0),
}

def llm_energy_wh(provider: str, tokens_in: int, tokens_out: int,
                  p_gpu_w: float = 0.0, t_inference_s: float = 0.0):
    """
    Calcule l'énergie d'inférence LLM en Wh [8][9].
    Retourne (e_nominal, e_min, e_max) pour reporter l'incertitude.

    Pour ollama avec mesure directe GPU (p_gpu_w > 0) : E = P_GPU * t [8].
    Sinon : E = tokens_in * eps_in + tokens_out * eps_out [8][9].
    """
    eps_in, eps_out, unc = LLM_ENERGY_PROFILES.get(provider, (0.005, 0.015, 5.0))
    if provider == "ollama" and p_gpu_w > 0 and t_inference_s > 0:
        e = p_gpu_w * t_inference_s / 3600.0
        return e, e * 0.8, e * 1.2
    e_nom = eps_in * tokens_in / 1000.0 + eps_out * tokens_out / 1000.0
    return e_nom, e_nom / unc, e_nom * unc


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
    # Métriques énergétiques — régime datastream (puissance continue [W])
    # Réf: Barroso [1], Ahvar [2], Ait Salaht [5]
    # p_node_w  : puissance nœuds actifs P(f) = Σ PUE*[P_idle+(P_max-P_idle)*u]  [1][2]
    # p_link_w  : proxy réseau Σ bw*lat  (normalisé, sans unité W stricte)        [5][6]
    # p_total_w : objectif de minimisation = p_node_w + p_link_w
    # energy_node/link/total : aliases pour compatibilité CSP externe (même valeur)
    p_node_w: float = 0.0
    p_link_w: float = 0.0
    p_total_w: float = 0.0
    energy_link: float = 0.0   # alias p_link_w
    energy_node: float = 0.0   # alias p_node_w
    energy_total: float = 0.0  # alias p_total_w
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
    e_solver_wh: float = 0.0       # énergie nominale [Wh]
    e_solver_wh_min: float = 0.0   # borne basse (incertitude [9])
    e_solver_wh_max: float = 0.0   # borne haute (incertitude [9])
    cost_usd: float = 0.0
    e_method: str = ""  # "P*T", "tok*(eps_in,eps_out)", "~0"


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



# =====================================================================
# CALCUL DU DIAMÈTRE ET MÉTRIQUES DU GRAPHE
# =====================================================================
def compute_graph_diameter_hops(infra) -> int:
    """
    Calcule le diamètre du graphe en nombre de SAUTS (hops).
    Utilise BFS depuis chaque nœud.
    
    Diamètre = max { distance(u, v) | pour tout u, v }
    où distance = nombre minimum de sauts (arcs)
    """
    n = infra.nbH
    
    # Construire liste d'adjacence simple (sans poids)
    adj_simple = defaultdict(list)
    for u in infra.adj:
        for (v, bw, lat) in infra.adj[u]:
            if u != v:
                adj_simple[u].append(v)
    
    diameter = 0
    
    # BFS depuis chaque nœud pour trouver les distances en sauts
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        
        while queue:
            u = queue.popleft()
            for v in adj_simple[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    if dist[v] > diameter:
                        diameter = dist[v]
    
    return diameter


def compute_eccentricities_hops(infra) -> Dict[int, int]:
    """
    Calcule l'excentricité de chaque nœud en nombre de sauts.
    excentricité(v) = max { distance(v, u) | pour tout u atteignable }
    """
    n = infra.nbH
    
    adj_simple = defaultdict(list)
    for u in infra.adj:
        for (v, bw, lat) in infra.adj[u]:
            if u != v:
                adj_simple[u].append(v)
    
    eccentricities = {}
    
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        max_dist = 0
        
        while queue:
            u = queue.popleft()
            for v in adj_simple[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    if dist[v] > max_dist:
                        max_dist = dist[v]
        
        eccentricities[src] = max_dist
    
    return eccentricities


def print_graph_metrics(infra):
    """Affiche les métriques du graphe : diamètre, rayon, centre."""
    print("\n" + "=" * 60)
    print(" METRIQUES DU GRAPHE D INFRASTRUCTURE")
    print("=" * 60)
    
    # Calculer le diamètre
    computed_diameter = compute_graph_diameter_hops(infra)
    eccentricities = compute_eccentricities_hops(infra)
    
    # Rayon = excentricité minimale
    radius = min(eccentricities.values()) if eccentricities else 0
    
    # Centre = nœuds avec excentricité = rayon
    center_nodes = [v for v, e in eccentricities.items() if e == radius]
    
    # Périphérie = nœuds avec excentricité = diamètre
    periphery_nodes = [v for v, e in eccentricities.items() if e == computed_diameter]
    
    # Compter les arcs (sans boucles)
    num_arcs = sum(1 for u in infra.adj for (v, _, _) in infra.adj[u] if u != v)
    
    print(f"  Nombre de noeuds        : {infra.nbH}")
    print(f"  Nombre d arcs           : {num_arcs} ({num_arcs // 2} bidirectionnels)")
    print(f"  Diametre (fichier)      : {infra.diameter} sauts")
    print(f"  Diametre (calcule)      : {computed_diameter} sauts")
    
    if infra.diameter != computed_diameter:
        effective = infra.diameter #min(infra.diameter, computed_diameter)
        print(f"  >> ATTENTION: Le diametre du fichier ({infra.diameter}) "
              f"differe du diametre calcule ({computed_diameter})!")
        print(f"  >> Correction : infra.diameter <- min({infra.diameter}, {computed_diameter}) = {effective}")
        infra.diameter = effective
    
    print(f"  Rayon                   : {radius} sauts")
    print(f"  Centre ({len(center_nodes)} noeuds)       : {center_nodes[:8]}{'...' if len(center_nodes) > 8 else ''}")
    print(f"  Peripherie ({len(periphery_nodes)} noeuds)   : {periphery_nodes[:8]}{'...' if len(periphery_nodes) > 8 else ''}")
    
    # Distribution des excentricités
    ecc_dist = Counter(eccentricities.values())
    print(f"  Distribution excentricites : {dict(sorted(ecc_dist.items()))}")
    print("=" * 60)
    
    return computed_diameter


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


def validate_dz_against_infra(appli, infra):
    """Vérifie que toutes les contraintes DZ référencent des nœuds valides.
    Lève une ValueError explicite si un hôte DZ est hors de l'infrastructure."""
    errors = []
    for ci, hi in appli.DZ:
        if hi >= infra.nbH:
            errors.append(
                f"  DZ: composant C{ci} assigné à H{hi}, "
                f"mais l'infra n'a que {infra.nbH} nœuds (H0..H{infra.nbH-1})."
            )
        if ci >= appli.nbC:
            errors.append(
                f"  DZ: composant C{ci} hors des {appli.nbC} composants de l'application."
            )
    if errors:
        msg = ("\nERREUR — Incompatibilité DZ / infrastructure :\n" +
               "\n".join(errors) +
               "\n\nSolutions possibles :"
               "\n  1. Utiliser une infrastructure avec au moins "
               f"{max(hi for _, hi in appli.DZ) + 1} nœuds."
               "\n  2. Modifier component.DZ dans le fichier application "
               "pour référencer un nœud existant."
               "\n  3. Utiliser --infra avec une topologie compatible.")
        raise ValueError(msg)


def compute_norm_bounds(infra, appli):
    b = NormBounds()
    max_arc_lat = max(infra.arc_lat.values(), default=1)
    # Diamètre en latence réelle = max(sp_latency[u][v]) sur toutes paires atteignables
    # Plus précis que diameter_hops × max_arc_lat quand les latences sont hétérogènes
    lat_diameter = max(
        (infra.sp_latency[u][v]
         for u in range(infra.nbH)
         for v in range(infra.nbH)
         if infra.sp_latency[u][v] < INF_V),
        default=infra.diameter * max_arc_lat
    )
    b.lat_ub = max(1, appli.nbL * lat_diameter)
    b.cost_max = max(1, sum(infra.hostCost))
    max_flow = max(1, sum(appli.bdwPair))
    # P_link borne haute : proxy linéaire bw*lat [5][6] (régime datastream)
    b.e_link_ub = max(1, max_flow * lat_diameter * GCP_FACTOR_CROSSREGION)
    # P_node borne haute : tous nœuds à pleine charge, modèle vCPU GCP [2][4]
    # P_max(h) = GCP_PUE * P_vcpu_active * cpu_cap
    max_p_eff = max(
        GCP_PUE * GCP_P_VCPU_ACTIVE_W * c
        for c in infra.cpuCap
    )
    b.e_node_ub = max(1, int(infra.nbH * max_p_eff))
    b.energy_ub = b.e_link_ub + b.e_node_ub
    return b


# =====================================================================
# EVALUATEUR UNIFIE (utilise par toutes les approches sauf CSP)
# =====================================================================
def evaluate_solution(placement, infra, appli, bounds,
                      w_lat=0.34, w_cost=0.33, w_energy=0.33):
    sol = PlacementSolution(); sol.placement = dict(placement)
    sol.violations = []

    # C0: Completude
    for c in range(appli.nbC):
        if c not in placement:
            sol.violations.append(f"C0: Composant {c} non place")
            sol.placement[c] = 0

    # C1: Localite DZ
    for ci, hi in appli.DZ:
        if hi >= infra.nbH:
            sol.violations.append(f"C1: DZ H{hi} hors infra (nbH={infra.nbH})")
            continue
        if sol.placement.get(ci) != hi:
            sol.violations.append(f"C1: DZ C{ci} devrait etre sur H{hi}, est sur H{sol.placement.get(ci)}")

    # C2: Capacite
    cu = [0] * infra.nbH; ru = [0] * infra.nbH
    for c in range(appli.nbC):
        h = sol.placement.get(c, 0)
        if h >= infra.nbH:
            sol.violations.append(f"C2: Composant C{c} placé sur H{h} hors infra")
            h = 0
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

    # C5: Latence maximale par lien applicatif
    # Contrainte de QoS : la latence bout-en-bout de chaque lien l
    # doit respecter latPair[l] si défini (>0).
    for l in range(appli.nbL):
        max_lat = appli.latPair[l]
        if max_lat > 0 and sol.consumed_lat.get(l, 0) > max_lat:
            sol.violations.append(
                f"C5: Lat L{l} (C{appli.linkPerServ[l][0]}→C{appli.linkPerServ[l][1]}): "
                f"{sol.consumed_lat[l]}>{max_lat}ms")

    # Metriques
    sol.worst_case_latency = max(link_lats) if link_lats else 0
    sol.total_latency = sum(x for x in link_lats if x < INF_V)
    sol.infra_cost = sum(infra.hostCost[h] for h in sol.active_hosts)
    # Puissance continue — régime datastream [1][2][5][6]
    # p_link_w : proxy linéaire bw*lat, normalisé (pas d'unité W stricte) [5][6]
    sol.p_link_w = sum(
        link_energy_proxy(fl, infra.arc_lat.get((u, v), 0))
        for (u, v), fl in arc_flow.items())
    # p_node_w : modèle Barroso [1] étendu continuum [2], tier-aware
    sol.p_node_w = sum(
        node_power_w(cu[h], infra.cpuCap[h])
        for h in sol.active_hosts)
    sol.p_total_w = sol.p_link_w + sol.p_node_w
    # Aliases pour compatibilité CSP externe et affichage
    sol.energy_link  = sol.p_link_w
    sol.energy_node  = sol.p_node_w
    sol.energy_total = sol.p_total_w
    sol.feasible = len(sol.violations) == 0

    # Objectif normalise (SOMME bout-en-bout dans le terme latence)
    if sol.feasible:
        def norm_int(val, ub):
            return (val * NORM) // ub if ub > 0 else 0
        lat_n = norm_int(sol.total_latency, bounds.lat_ub)
        cost_n = norm_int(sol.infra_cost, bounds.cost_max)
        ener_n = norm_int(sol.energy_total, bounds.energy_ub)  # = p_total_w normalisé
        W = 1000
        wl, wc, we = int(round(w_lat * W)), int(round(w_cost * W)), int(round(w_energy * W))
        sol.objective_int = lat_n * wl + cost_n * wc + ener_n * we
        sol.objective = float(sol.objective_int)
    else:
        sol.objective_int = 10 ** 9; sol.objective = float("inf")
    return sol


# =====================================================================
# GLOUTON ENERGY-AWARE (FFD) — objectif: min ΔE_total = ΔE_nodes + ΔE_links
# =====================================================================
# =====================================================================
# UTILITAIRES GREEDY — BW-aware, génériques (toute infra/appli)
# =====================================================================

def _comp_link_index(appli):
    """Index des liens par composant — mémoïsé. Générique."""
    if not hasattr(appli, '_comp_links_cache'):
        idx = defaultdict(list)
        for l in range(appli.nbL):
            sC, tC = appli.linkPerServ[l]
            idx[sC].append((tC, appli.bdwPair[l], l))
            idx[tC].append((sC, appli.bdwPair[l], l))
        appli._comp_links_cache = idx
    return appli._comp_links_cache


def _arc_flow_from_placement(pl, infra, appli):
    """Calcule le flux courant sur chaque arc pour un placement (partiel ou total)."""
    arc_flow = defaultdict(int)
    for l in range(appli.nbL):
        sC, tC = appli.linkPerServ[l]
        if sC not in pl or tC not in pl:
            continue
        hs, ht = pl[sC], pl[tC]
        if hs == ht:
            continue
        path = infra.sp_path.get((hs, ht), [])
        for i in range(len(path) - 1):
            arc_flow[(path[i], path[i+1])] += appli.bdwPair[l]
    return arc_flow


def _extra_flow_for(h, c, pl, infra, appli):
    """Retourne le dict {arc: BW_ajoutée} si on place c sur h.
    Respecte la DIRECTION réelle du flux réseau selon le sens du lien applicatif.
    Le flux de sC@h_src → tC@h_dst emprunte sp_path(h_src, h_dst).
    Générique quel que soit le fichier application ou infra."""
    extra = defaultdict(int)
    for nb, bw, l_idx in _comp_link_index(appli)[c]:
        if nb not in pl:
            continue
        h_nb = pl[nb]
        if h == h_nb:
            continue  # co-location : flux nul
        # Déterminer la direction réelle du flux selon le lien applicatif
        sC, tC = appli.linkPerServ[l_idx]
        if sC == c:
            # c est la SOURCE : flux de h (c) vers h_nb (nb)
            h_src, h_dst = h, h_nb
        else:
            # c est la DESTINATION : flux de h_nb (nb) vers h (c)
            h_src, h_dst = h_nb, h
        path = infra.sp_path.get((h_src, h_dst), [])
        if not path:
            extra[(-1, -1)] = INF_V
            return extra
        for i in range(len(path) - 1):
            extra[(path[i], path[i + 1])] += bw
    return extra


def _bw_feasible(h, c, pl, infra, appli, arc_flow_used):
    """Vérifie que placer c sur h ne sature aucun arc.
    Cumule tout le flux ajouté par c→h AVANT de comparer aux capacités.
    Générique : fonctionne quel que soit le DZ ou la topologie."""
    extra = _extra_flow_for(h, c, pl, infra, appli)
    for arc, added in extra.items():
        if arc == (-1, -1):
            return False  # pas de chemin
        cap = infra.arc_bw.get(arc, 0)
        if cap > 0 and arc_flow_used[arc] + added > cap:
            return False
    return True


def _commit_placement(c, h, pl, cu, ru, infra, appli, arc_flow_used):
    """Place c sur h et met à jour CPU, RAM et arc_flow_used."""
    pl[c] = h
    cu[h] += appli.cpuComp[c]
    ru[h] += appli.ramComp[c]
    for arc, bw in _extra_flow_for(h, c, pl, infra, appli).items():
        if arc != (-1, -1):
            arc_flow_used[arc] += bw


def _bw_repair_pass(pl, cu, ru, infra, appli, arc_flow_used, bounds):
    """Phase 2 — Repair pass BW générique.

    Stratégie améliorée :
      Pour chaque arc saturé (par ordre de saturation décroissant) :
        A) Pour chaque composant contributeur (non-DZ), tentative de déplacement
           vers un hôte qui CONTOURNE l'arc saturé (BW-faisable global).
        B) Si échec, tentative de CO-LOCALISATION avec un composant déjà placé
           (flux inter-hôte = 0 → toujours BW-safe sur l'arc saturé).
        C) Si toujours bloqué, tentative de déplacement de PAIRES de composants
           communiquant via l'arc saturé (co-localisation forcée des deux).
      Répété jusqu'à convergence ou max_iter.

    Ne modifie jamais les composants DZ.
    Générique : aucune hypothèse sur la topologie ou l'application.
    """
    dz_comps = {ci for ci, _ in appli.DZ}
    max_iter = appli.nbC * infra.nbH  # borne large
    link_idx = _comp_link_index(appli)

    for iteration in range(max_iter):
        # Recalculer flux courant
        arc_flow_used.clear()
        for arc, bw in _arc_flow_from_placement(pl, infra, appli).items():
            arc_flow_used[arc] += bw

        saturated = {arc: flow for arc, flow in arc_flow_used.items()
                     if infra.arc_bw.get(arc, 0) > 0
                     and flow > infra.arc_bw[arc]}
        if not saturated:
            break

        fixed_any = False

        # Trier par saturation décroissante (attaquer le pire arc en premier)
        for sat_arc, sat_flow in sorted(saturated.items(),
                                        key=lambda x: x[1] - infra.arc_bw.get(x[0], 1),
                                        reverse=True):
            cap = infra.arc_bw.get(sat_arc, INF_V)

            # — Identifier les composants dont le flux passe par sat_arc —
            contributors = []  # (comp, bw_contrib, lien_idx)
            for l in range(appli.nbL):
                sC, tC = appli.linkPerServ[l]
                if sC not in pl or tC not in pl: continue
                hs, ht = pl[sC], pl[tC]
                if hs == ht: continue
                path = infra.sp_path.get((hs, ht), [])
                for i in range(len(path) - 1):
                    if (path[i], path[i + 1]) == sat_arc:
                        bw_l = appli.bdwPair[l]
                        for comp in (sC, tC):
                            if comp not in dz_comps:
                                contributors.append((comp, bw_l, l))
                        break

            # Dédupliquer, trier par BW contributée décroissante
            seen_c = {}
            for comp, bw_l, l in contributors:
                seen_c[comp] = seen_c.get(comp, 0) + bw_l
            candidates = sorted(seen_c.items(), key=lambda x: x[1], reverse=True)

            # — Tentative A : déplacer chaque candidat vers un hôte contournant sat_arc —
            for (c_move, _) in candidates:
                old_h = pl[c_move]
                cu[old_h] -= appli.cpuComp[c_move]
                ru[old_h] -= appli.ramComp[c_move]
                del pl[c_move]
                af_tmp = _arc_flow_from_placement(pl, infra, appli)

                best_h = -1; best_dp = float('inf')
                for h in range(infra.nbH):
                    if cu[h] + appli.cpuComp[c_move] > infra.cpuCap[h]: continue
                    if ru[h] + appli.ramComp[c_move] > infra.ramCap[h]: continue
                    if not _bw_feasible(h, c_move, pl, infra, appli, af_tmp): continue
                    # C5 : vérifier latence max pour les liens du composant déplacé
                    lat_ok_r = True
                    for nb_r, bw_r, l_idx_r in link_idx[c_move]:
                        if nb_r not in pl: continue
                        max_lat_r = appli.latPair[l_idx_r]
                        if max_lat_r > 0 and infra.sp_latency[h][pl[nb_r]] > max_lat_r:
                            lat_ok_r = False; break
                    if not lat_ok_r: continue
                    # Vérifier que sat_arc n'est plus saturé avec ce nouveau placement
                    extra = _extra_flow_for(h, c_move, pl, infra, appli)
                    new_sat_flow = af_tmp.get(sat_arc, 0) + extra.get(sat_arc, 0)
                    if new_sat_flow > cap: continue
                    dp = (node_power_w(cu[h] + appli.cpuComp[c_move], infra.cpuCap[h])
                          - node_power_w(cu[h], infra.cpuCap[h]))
                    for nb, bw_l, _ in link_idx[c_move]:
                        if nb not in pl: continue
                        h_nb = pl[nb]
                        if h != h_nb:
                            lat = infra.sp_latency[h][h_nb]
                            dp += link_energy_proxy(bw_l, lat) if lat < INF_V else bw_l * 9999
                    if dp < best_dp:
                        best_dp = dp; best_h = h

                if best_h >= 0 and best_h != old_h:
                    _commit_placement(c_move, best_h, pl, cu, ru, infra, appli, af_tmp)
                    arc_flow_used.clear()
                    arc_flow_used.update(af_tmp)
                    fixed_any = True
                    break
                else:
                    # Remettre en place
                    _commit_placement(c_move, old_h, pl, cu, ru, infra, appli,
                                      defaultdict(int, arc_flow_used))

            if fixed_any:
                break  # Recommencer depuis le début

            # — Tentative B : co-localiser chaque candidat avec un composant déjà placé —
            # La co-localisation annule le flux entre les deux → toujours BW-safe
            for (c_move, _) in candidates:
                old_h = pl[c_move]
                cu[old_h] -= appli.cpuComp[c_move]
                ru[old_h] -= appli.ramComp[c_move]
                del pl[c_move]

                best_coloc_h = -1; best_coloc_dp = float('inf')
                # Essayer tous les hôtes déjà occupés (co-localisation large)
                occupied = set(pl.values())
                for h in occupied:
                    if cu[h] + appli.cpuComp[c_move] > infra.cpuCap[h]: continue
                    if ru[h] + appli.ramComp[c_move] > infra.ramCap[h]: continue
                    af_tmp = _arc_flow_from_placement(pl, infra, appli)
                    if not _bw_feasible(h, c_move, pl, infra, appli, af_tmp): continue
                    extra = _extra_flow_for(h, c_move, pl, infra, appli)
                    new_sat_flow = af_tmp.get(sat_arc, 0) + extra.get(sat_arc, 0)
                    if new_sat_flow > cap: continue
                    dp = (node_power_w(cu[h] + appli.cpuComp[c_move], infra.cpuCap[h])
                          - node_power_w(cu[h], infra.cpuCap[h]))
                    if dp < best_coloc_dp:
                        best_coloc_dp = dp; best_coloc_h = h

                if best_coloc_h >= 0 and best_coloc_h != old_h:
                    af_tmp = _arc_flow_from_placement(pl, infra, appli)
                    _commit_placement(c_move, best_coloc_h, pl, cu, ru, infra, appli, af_tmp)
                    arc_flow_used.clear()
                    arc_flow_used.update(af_tmp)
                    fixed_any = True
                    break
                else:
                    _commit_placement(c_move, old_h, pl, cu, ru, infra, appli,
                                      defaultdict(int, arc_flow_used))

            if fixed_any:
                break

            # — Tentative C : co-localiser des composants contribuant à sat_arc
            #   sur un hôte quelconque (pas seulement h_fixed).
            #   Exploration de tous les hôtes occupés pour chaque candidat.
            #   Priorité : hôtes déjà occupés (évite d'allumer un nouveau nœud). —
            for (c_move, bw_contrib) in candidates:
                old_h = pl[c_move]
                cu[old_h] -= appli.cpuComp[c_move]
                ru[old_h] -= appli.ramComp[c_move]
                del pl[c_move]

                best_coloc_h = -1; best_coloc_dp = float('inf')
                # Essayer tous les hôtes (occupés d'abord, puis libres)
                occupied_sorted = sorted(set(pl.values()),
                                         key=lambda h: node_power_w(cu[h], infra.cpuCap[h]))
                all_h_sorted = occupied_sorted + [
                    h for h in sorted(range(infra.nbH), key=lambda h: -infra.cpuCap[h])
                    if h not in set(pl.values())
                ]
                for h in all_h_sorted:
                    if cu[h] + appli.cpuComp[c_move] > infra.cpuCap[h]: continue
                    if ru[h] + appli.ramComp[c_move] > infra.ramCap[h]: continue
                    af_tmp = _arc_flow_from_placement(pl, infra, appli)
                    if not _bw_feasible(h, c_move, pl, infra, appli, af_tmp): continue
                    extra = _extra_flow_for(h, c_move, pl, infra, appli)
                    new_sat_flow = af_tmp.get(sat_arc, 0) + extra.get(sat_arc, 0)
                    if new_sat_flow > cap: continue
                    dp = (node_power_w(cu[h] + appli.cpuComp[c_move], infra.cpuCap[h])
                          - node_power_w(cu[h], infra.cpuCap[h]))
                    if dp < best_coloc_dp:
                        best_coloc_dp = dp; best_coloc_h = h
                        break  # premier hôte valide suffit (optimisation)

                if best_coloc_h >= 0 and best_coloc_h != old_h:
                    af_tmp = _arc_flow_from_placement(pl, infra, appli)
                    _commit_placement(c_move, best_coloc_h, pl, cu, ru, infra, appli, af_tmp)
                    arc_flow_used.clear()
                    arc_flow_used.update(af_tmp)
                    fixed_any = True
                    break
                else:
                    _commit_placement(c_move, old_h, pl, cu, ru, infra, appli,
                                      defaultdict(int, arc_flow_used))

            if fixed_any:
                break

        if not fixed_any:
            break  # Aucun progrès possible sur aucun arc


def greedy_energy_placement(infra, appli):
    """Greedy FFD énergie-aware avec repair BW multi-stratégie — générique.

    Respecte TOUTES les contraintes, quel que soit le fichier infra ou application :
      C1 - Localité DZ     (placement prioritaire, aucune hypothèse sur les hôtes DZ)
      C2 - Capacité CPU/RAM (filtrage systématique)
      C3 - Connectivité    (chemin SP requis)
      C4 - Bande passante  (score BW-aware Phase 1 + repair Phase 2 multi-stratégie)

    Architecture 2 phases :
      Phase 1 — Construction FFD BW-aware :
        Tri décroissant CPU. Score = ΔP_node + ΔP_link + pénalité BW saturée.
        Si aucun hôte BW-faisable : fallback co-localisation voisin BW-safe.

      Phase 2 — Repair pass BW (3 stratégies) :
        A) Déplacement simple vers hôte contournant l'arc saturé
        B) Co-localisation large avec tout hôte déjà occupé
        C) Co-localisation de paires de composants communiquant via l'arc saturé
        → Ne modifie jamais les composants DZ.

    Garantit la faisabilité si le problème est structurellement faisable,
    pour toute topologie et toute application.
    """
    # Valider DZ — générique
    for ci, hi in appli.DZ:
        if hi >= infra.nbH:
            raise IndexError(
                f"DZ C{ci}→H{hi} hors de l'infra ({infra.nbH} nœuds). "
                f"Utiliser validate_dz_against_infra() avant l'appel."
            )

    pl = {}
    cu = [0] * infra.nbH
    ru = [0] * infra.nbH
    arc_flow_used = defaultdict(int)
    link_idx = _comp_link_index(appli)

    # Phase 1a : placer les DZ en premier
    for ci, hi in appli.DZ:
        _commit_placement(ci, hi, pl, cu, ru, infra, appli, arc_flow_used)

    # Phase 1b : FFD BW-aware sur les composants libres
    # Tri : CPU décroissant (FFD), à égalité : nombre de liens décroissant
    def sort_key(c):
        return (-appli.cpuComp[c], -len(link_idx[c]))
    free = sorted([c for c in range(appli.nbC) if c not in pl], key=sort_key)

    for c in free:
        best_h = -1
        best_dp = float('inf')

        for h in range(infra.nbH):
            if cu[h] + appli.cpuComp[c] > infra.cpuCap[h]: continue
            if ru[h] + appli.ramComp[c] > infra.ramCap[h]: continue
            if not _bw_feasible(h, c, pl, infra, appli, arc_flow_used): continue

            # C5 : Vérifier la contrainte de latence max pour les liens déjà placés
            lat_ok = True
            for nb, bw, l_idx in link_idx[c]:
                if nb not in pl: continue
                max_lat = appli.latPair[l_idx]
                if max_lat > 0:
                    lat_hn = infra.sp_latency[h][pl[nb]]
                    if lat_hn > max_lat:
                        lat_ok = False; break
            if not lat_ok: continue

            # ΔP_node : Barroso [1][2]
            dp = (node_power_w(cu[h] + appli.cpuComp[c], infra.cpuCap[h])
                  - node_power_w(cu[h], infra.cpuCap[h]))
            # ΔP_link : proxy linéaire [5][6]
            for nb, bw, _ in link_idx[c]:
                if nb not in pl: continue
                h_nb = pl[nb]
                if h != h_nb:
                    lat = infra.sp_latency[h][h_nb]
                    dp += link_energy_proxy(bw, lat) if lat < INF_V else bw * 9999

            if dp < best_dp:
                best_dp = dp; best_h = h

        if best_h < 0:
            # Fallback A : co-localisation avec un voisin déjà placé
            # (flux inter-hôte = 0 → toujours BW-safe)
            for nb, bw, _ in sorted(link_idx[c], key=lambda x: -x[1]):
                if nb not in pl: continue
                h_nb = pl[nb]
                if cu[h_nb] + appli.cpuComp[c] > infra.cpuCap[h_nb]: continue
                if ru[h_nb] + appli.ramComp[c] > infra.ramCap[h_nb]: continue
                best_h = h_nb
                break

        if best_h < 0:
            # Fallback B : co-localisation avec n'importe quel hôte occupé (CPU/RAM ok)
            for h in sorted(set(pl.values()),
                            key=lambda h: node_power_w(cu[h], infra.cpuCap[h])):
                if cu[h] + appli.cpuComp[c] <= infra.cpuCap[h] and                    ru[h] + appli.ramComp[c] <= infra.ramCap[h]:
                    best_h = h; break

        if best_h < 0:
            # Fallback C : VM à grande capacité (gcp-large en tête)
            for h in sorted(range(infra.nbH),
                            key=lambda h: -infra.cpuCap[h]):
                if cu[h] + appli.cpuComp[c] <= infra.cpuCap[h] and                    ru[h] + appli.ramComp[c] <= infra.ramCap[h]:
                    best_h = h; break

        if best_h >= 0:
            _commit_placement(c, best_h, pl, cu, ru, infra, appli, arc_flow_used)
        # Sinon : infaisable structurel (CPU/RAM insuffisant sur toute l'infra)

    # Phase 2 : repair pass BW multi-stratégie
    _bw_repair_pass(pl, cu, ru, infra, appli, arc_flow_used, None)

    return pl

def greedy_placement_legacy(infra, appli):
    """Greedy legacy : score = hostCost×10 + Σ latence. BW-aware via repair pass."""
    pl = {}; cu = [0] * infra.nbH; ru = [0] * infra.nbH
    arc_flow_used = defaultdict(int)
    for ci, hi in appli.DZ:
        if hi >= infra.nbH:
            raise IndexError(f"DZ C{ci}→H{hi} hors de l'infra ({infra.nbH} nœuds).")
        _commit_placement(ci, hi, pl, cu, ru, infra, appli, arc_flow_used)
    for c in range(appli.nbC):
        if c in pl: continue
        bh = -1; bs = float("inf")
        for h in range(infra.nbH):
            if cu[h] + appli.cpuComp[c] > infra.cpuCap[h]: continue
            if ru[h] + appli.ramComp[c] > infra.ramCap[h]: continue
            cost_pen = infra.hostCost[h] if cu[h] == 0 else 0
            lat_pen = sum(
                infra.sp_latency[h][pl[nb]] if infra.sp_latency[h][pl[nb]] < INF_V else 9999
                for nb in pl
                for l in range(appli.nbL)
                if appli.linkPerServ[l] in ((c, nb), (nb, c))
            )
            sc = cost_pen * 10 + lat_pen
            if sc < bs: bs = sc; bh = h
        if bh >= 0:
            _commit_placement(c, bh, pl, cu, ru, infra, appli, arc_flow_used)
        else:
            pl[c] = 0
    # Repair pass BW
    _bw_repair_pass(pl, cu, ru, infra, appli, arc_flow_used, None)
    return pl

greedy_placement = greedy_energy_placement


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
        D4 = max(1, infra.diameter // 4)
        k = max(6, sum(1 for h in range(infra.nbH) if infra.sp_latency[dh][h] <= D4))
        k = min(k, infra.nbH // 4)
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

    # FIX v9: pré-calculer les top hôtes de consolidation une seule fois (hors _candidate_hosts)
    _bw_dz_total_ls = 0
    for l in range(appli.nbL):
        sc, tc = appli.linkPerServ[l]
        if (sc in dz_map) != (tc in dz_map):  # exactement l'un est DZ
            _bw_dz_total_ls += appli.bdwPair[l]
    _total_free_cpu_ls = sum(appli.cpuComp[c] for c in range(appli.nbC) if c not in dz_map)

    # FIX v10-gcp: trier par zone GCP (intra < inter < cross) puis par coût réel (el+en).
    # Cohérent avec la logique consol_rank de run_llm() :
    #   Zone 1 intra (≤2ms)  > zone 2 inter (≤25ms) > zone 3 cross (>25ms)
    #   Au sein d'une zone : el+en détermine le meilleur nœud.
    _dz_h_main_ls = list(dz_hosts)[0] if dz_hosts else 0
    top_consol_hosts_ls = sorted(
        [h for h in range(infra.nbH)
         if h not in set(dz_hosts)
         and all(infra.sp_latency[dh][h] < INF_V for dh in dz_hosts)],
        key=lambda h: (
            # Clé 1 : zone tier (0=DZ, 1=intra, 2=inter, 3=cross)
            (0 if infra.sp_latency[_dz_h_main_ls][h] == 0
             else 1 if infra.sp_latency[_dz_h_main_ls][h] <= GCP_LAT_INTRAZONE_MS
             else 2 if infra.sp_latency[_dz_h_main_ls][h] <= GCP_LAT_INTERZONE_MS
             else 3),
            # Clé 2 : énergie lien + énergie nœud dans la zone
            _bw_dz_total_ls * infra.sp_latency[_dz_h_main_ls][h]
            + node_power_w(_total_free_cpu_ls, infra.cpuCap[h])
        )
    )[:10]  # 10 meilleurs hôtes : intra-zone en priorité, puis inter, puis cross

    def _candidate_hosts(comp):
        """Hôtes actifs + DZ + voisins + rayon D/4 depuis DZ + top consolidation hosts.
        FIX v9: les hôtes de consolidation optimaux sont toujours inclus même hors D/4."""
        cands = set()
        for h in range(infra.nbH):
            if cu[h] > 0: cands.add(h)
        for dh in dz_hosts: cands.add(dh)
        for nb, _ in comp_graph[comp]:
            cands.add(curr_pl[nb])
        cands.update(near_dz_hosts)
        # Rayon D/4 depuis chaque DZ host
        D4 = max(1, infra.diameter // 4)
        for dh in dz_hosts:
            for h in range(infra.nbH):
                if infra.sp_latency[dh][h] <= D4:
                    cands.add(h)
        # FIX v9: toujours inclure les meilleurs hôtes de consolidation globaux
        cands.update(top_consol_hosts_ls)
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
                            sc += bw * (d if d < INF_V else 999999)  # linear bw*lat [Ait Salaht 2019]
                        if lns_cu[h] == 0 and h not in set(dz_map.values()):
                            sc += GCP_PUE * GCP_P_VCPU_IDLE_W * infra.cpuCap[h]  # idle penalty GCP [4]
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
    csp_file = os.path.join(os.path.dirname(__file__), "Code_placement_csp_vf_v8.py")
    if not os.path.exists(csp_file):
        # fallback sur la version originale
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
        #solver.parameters.max_time_in_seconds = 120

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
        # Récupération depuis le solveur CSP externe [5]
        # Le modèle CSP utilise les mêmes formules que evaluate_solution
        # (proxy linéaire E_link, modèle Barroso E_node) via build_objective()
        # total_link_energy est en W (converti en cW dans build_objective via ×100)
        # total_node_energy est en cW (pue10 × p_W10 = W×100)
        # → diviser par 100 pour obtenir les valeurs en Watts cohérentes avec evaluate_solution()
        ps.p_link_w  = float(solver.Value(sol.total_link_energy))           # W brut
        ps.p_node_w  = float(solver.Value(sol.total_node_energy)) / 100.0   # cW → W
        ps.p_total_w = float(solver.Value(sol.total_energy_global)) / 100.0  # cW → W
        # Aliases pour cohérence avec evaluate_solution
        ps.energy_link  = ps.p_link_w
        ps.energy_node  = ps.p_node_w
        ps.energy_total = ps.p_total_w
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

        e_solver_wh = P_CPU_SOLVER_W * t_total / 3600.0
        stat = solver.StatusName(status)
        if verbose:
            print(f"  [CSP] {stat} | Obj={ps.objective:.0f} Lat_e2e={ps.total_latency}ms "
                  f"P_total={ps.p_total_w:.2f}W | t={t_total:.2f}s E_solver={e_solver_wh:.6f}Wh")

        return SolverResult(
            name=f"CSP ({stat})", sol=ps,
            t_total_s=t_total, e_solver_wh=e_solver_wh,
            e_solver_wh_min=e_solver_wh * 0.8, e_solver_wh_max=e_solver_wh * 1.2,
            e_method=f"P*T ({P_CPU_SOLVER_W:.0f}W) [locale]")
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
LLM_SYSTEM = """You are an expert optimizer for the Service Placement Problem (SPP) on Google Cloud Platform (GCP).
The infrastructure consists exclusively of GCE VMs (e2/n2/c2 families). There are NO fog nodes, NO edge devices.
Your goal is to find a placement that MINIMIZES the ENERGY objective. Lower is better.

OBJECTIVE FUNCTION (ENERGY ONLY):
  obj = P_total = P_links + P_nodes  (continuous power [W], datastream regime)

WHERE:
- P_links = SUM over all application links l (GCP-aware model [Ait Salaht 2019, extended]):
    bandwidth[l] × sp_latency(hs_l, ht_l) × factor(link_type)
  Link type inferred from measured latency [GCP network measurements]:
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
Below you will see PREVIOUS placement attempts with their objective scores, sorted from worst to best.
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
# A2-A5: ENHANCED LLM SOLVER
# =====================================================================
def _build_problem_description(infra, appli, bounds, w, max_hosts_detail=50):
    """Construit une description structurée du problème pour le LLM.
    
    Inclut une analyse topologique basée sur:
    - Le diamètre du graphe d'infrastructure
    - Les distances optimales pour chaque lien d'application
    - Les chemins optimaux entre paires de composants
    """
    lines = []

    # =========================================================================
    # 1. INFRASTRUCTURE GCP SUMMARY
    # =========================================================================
    lines.append("=== GCP INFRASTRUCTURE ===")
    lines.append(f"VMs: {infra.nbH} GCE instances | Diameter: {infra.diameter} hops | PUE=1.10 (uniform) [Google Env. Report 2023]")
    lines.append("Network model: E_link = bw × sp_lat × factor  [factor: ≤2ms→1.0, ≤25ms→1.5, >25ms→2.0]")
    lines.append("")

    # Inférer les zones GCP depuis les latences au nœud H0
    dz_anchor = 0  # H0 = DZ anchor (Kafka-Source fixé)
    zones: dict = {}
    for h in range(infra.nbH):
        lat = infra.sp_latency[dz_anchor][h]
        if lat == 0:
            zone_label = "zone-local"
        elif lat <= GCP_LAT_INTRAZONE_MS:
            zone_label = "zone-local"
        elif lat <= GCP_LAT_INTERZONE_MS:
            # Regrouper par plage de latence inter-zone
            zone_label = f"zone-region(~{lat}ms)"
        else:
            zone_label = f"region-remote(~{lat}ms)"
        zones.setdefault(zone_label, []).append(h)

    # Consolidation vers étiquettes propres (regroupement par plage)
    zone_groups: dict = {}
    for h in range(infra.nbH):
        lat = infra.sp_latency[dz_anchor][h]
        if lat <= GCP_LAT_INTRAZONE_MS:
            zl = "zone-local"
        elif lat <= GCP_LAT_INTERZONE_MS:
            zl = f"inter-zone(~{lat}ms)"
        else:
            zl = f"cross-region(~{lat}ms)"
        zone_groups.setdefault(zl, []).append(h)

    lines.append("GCP topology (inferred from measured latencies from H0):")
    for zlabel, hosts in sorted(zone_groups.items()):
        avg_cpu = sum(infra.cpuCap[h] for h in hosts) // max(1, len(hosts))
        largest_h = max(hosts, key=lambda h: infra.cpuCap[h])
        ltype, factor = infer_link_type(infra.sp_latency[dz_anchor][hosts[0]])
        lines.append(f"  {zlabel}: {len(hosts)} VMs (H{min(hosts)}..H{max(hosts)}) "
                     f"avg_cpu={avg_cpu} | link_factor={factor} [{ltype}] | "
                     f"largest: H{largest_h}({infra.cpuCap[largest_h]}vCPU)")

    lines.append("")
    lines.append("GCP ENERGY MODEL per VM:")
    lines.append(f"  P(h) = {GCP_PUE} × [{GCP_P_VCPU_IDLE_W}W×cpu_cap + ({GCP_P_VCPU_ACTIVE_W}-{GCP_P_VCPU_IDLE_W})W×cpu_used]")
    lines.append(f"  P_idle(h) = {GCP_PUE}×{GCP_P_VCPU_IDLE_W}×cpu_cap  |  P_max(h) = {GCP_PUE}×{GCP_P_VCPU_ACTIVE_W}×cpu_cap")
    lines.append("  Examples: e2-standard-2(2vCPU) idle=2.2W full=17.6W | n2-standard-8(8vCPU) idle=8.8W full=70.4W")
    lines.append("")
    lines.append("GCP NETWORK ENERGY MODEL:")
    lines.append("  E_link(l) = bw_l × sp_latency(hs,ht) × factor(type)  [5][6][7]")
    lines.append(f"  intra-zone  (≤{GCP_LAT_INTRAZONE_MS}ms)  : factor={GCP_FACTOR_INTRAZONE}  (local datacenter fabric)")
    lines.append(f"  inter-zone  (≤{GCP_LAT_INTERZONE_MS}ms) : factor={GCP_FACTOR_INTERZONE}  (backbone intra-region)")
    lines.append(f"  cross-region (>{GCP_LAT_INTERZONE_MS}ms) : factor={GCP_FACTOR_CROSSREGION}  (WAN long-haul — VERY EXPENSIVE)")
    lines.append(f"  Example: bw=100, lat=85ms(cross-region) → E=100×85×2.0=17000 vs lat=1ms(intra) → E=100×1×1.0=100")

    # =========================================================================
    # 2. APPLICATION DAG
    # =========================================================================
    lines.append("")
    lines.append("=== APPLICATION ===")
    lines.append(f"Components: {appli.nbC} | Links: {appli.nbL}")
    lines.append("")
    
    lines.append("Component requirements:")
    total_cpu = total_ram = 0
    for c in range(appli.nbC):
        total_cpu += appli.cpuComp[c]
        total_ram += appli.ramComp[c]
        lines.append(f"  C{c}: CPU={appli.cpuComp[c]}, RAM={appli.ramComp[c]}")
    lines.append(f"  TOTAL: CPU={total_cpu}, RAM={total_ram}")

    lines.append("")
    lines.append("Application links (dataflow graph):")
    for l, (s, t) in enumerate(appli.linkPerServ):
        lines.append(f"  L{l}: C{s} -> C{t}, bandwidth={appli.bdwPair[l]}, max_latency={appli.latPair[l]}ms")

    # DZ constraints
    dz_map = {ci: hi for ci, hi in appli.DZ}
    if appli.DZ:
        lines.append("")
        lines.append("DZ CONSTRAINTS (fixed placement):")
        for ci, hi in appli.DZ:
            lines.append(f"  C{ci} MUST be on H{hi} (CPU={infra.cpuCap[hi]}, tier={get_gcp_tier_label(infra.cpuCap[hi])})")

    # =========================================================================
    # 3. TOPOLOGICAL ANALYSIS — Diameter-based bounds
    # =========================================================================
    free_comps = [c for c in range(appli.nbC) if c not in dz_map]
    D = infra.diameter   # max shortest-path latency in the graph

    lines.append("")
    lines.append("=== GCP TOPOLOGICAL ANALYSIS — DIAMETER-BASED BOUNDS ===")
    lines.append(f"Graph diameter D = {D} ms  (worst-case shortest path between any two VMs)")
    lines.append(f"Any deployed path satisfies:  0 <= sp_latency(h_i, h_j) <= D = {D} ms")
    lines.append(f"Therefore for each app link l:  0 <= E_link(l) = bw_l * sp_lat * factor <= bw_l * D * {GCP_FACTOR_CROSSREGION}")
    lines.append("")
    lines.append("GCP ENERGY BOUNDS PER APPLICATION LINK (E_link = bw × sp_lat × factor):")
    lines.append(f"  {'Link':>5} {'Comp pair':>12} {'bw':>6}  E_worst(bw*D*2.0)  E_interzone(bw*D/2*1.5)  E_intra(bw*1ms*1.0)  E_opt(coloc)")

    total_e_worst = 0
    for l, (src_c, dst_c) in enumerate(appli.linkPerServ):
        bw = appli.bdwPair[l]
        e_worst = bw * D * GCP_FACTOR_CROSSREGION
        e_inter  = bw * (D // 2) * GCP_FACTOR_INTERZONE
        e_intra  = bw * 1 * GCP_FACTOR_INTRAZONE
        total_e_worst += e_worst
        lines.append(f"  L{l:>3}  C{src_c}->C{dst_c:>2}  {bw:>6}  {e_worst:>16.0f}  {e_inter:>22.0f}  {e_intra:>19.0f}  0 (coloc)")

    lines.append(f"  {'TOTAL':>5}  {'':>12}  {'':>6}  {total_e_worst:>16.0f}  (worst: all cross-region)")
    lines.append("")
    lines.append("GCP INTERPRETATION:")
    lines.append(f"  - SAME ZONE (≤{GCP_LAT_INTRAZONE_MS}ms): E_link = bw×lat×1.0  — cheapest, prefer always.")
    lines.append(f"  - SAME REGION (≤{GCP_LAT_INTERZONE_MS}ms): E_link = bw×lat×1.5  — acceptable.")
    lines.append(f"  - CROSS-REGION (>{GCP_LAT_INTERZONE_MS}ms): E_link = bw×lat×2.0 — AVOID: up to {int(GCP_FACTOR_CROSSREGION/GCP_FACTOR_INTRAZONE*D)}× more expensive.")
    lines.append(f"  - COLOCATE (same VM): E_link = 0  — always optimal if capacity allows.")
    lines.append("")

    lines.append("GCP LINK-BY-LINK CANDIDATE VMs (by zone proximity from fixed anchor):")
    lines.append("")

    for l, (src_c, dst_c) in enumerate(appli.linkPerServ):
        bw = appli.bdwPair[l]
        max_lat = appli.latPair[l]
        src_fixed = src_c in dz_map
        dst_fixed = dst_c in dz_map

        lines.append(f"  L{l}: C{src_c}->C{dst_c}  bw={bw}  max_lat={max_lat}ms")

        if src_fixed and dst_fixed:
            h_src, h_dst = dz_map[src_c], dz_map[dst_c]
            dist = infra.sp_latency[h_src][h_dst]
            ltype, factor = infer_link_type(dist)
            e_link = bw * dist * factor if dist < INF_V else float('inf')
            lines.append(f"    Both FIXED: H{h_src}->H{h_dst}, sp_lat={dist}ms [{ltype},×{factor}], E_link={e_link:.0f}")

        elif src_fixed or dst_fixed:
            anchor_c = src_c if src_fixed else dst_c
            free_c   = dst_c if src_fixed else src_c
            anchor_h = dz_map[anchor_c]
            cpu_need = appli.cpuComp[free_c]
            ram_need = appli.ramComp[free_c]

            lines.append(f"    C{anchor_c} FIXED on H{anchor_h} | C{free_c} FREE (CPU={cpu_need}, RAM={ram_need})")
            lines.append(f"    BEST: colocate C{free_c} on H{anchor_h} -> E_link=0  [check capacity]")

            for zone_label, zone_lat in [("intra-zone", GCP_LAT_INTRAZONE_MS),
                                          ("inter-zone", GCP_LAT_INTERZONE_MS),
                                          ("cross-region", D)]:
                _, factor = infer_link_type(zone_lat if zone_label != "cross-region" else zone_lat + 1)
                cands = []
                for h in range(infra.nbH):
                    d = infra.sp_latency[anchor_h][h]
                    lat_limit = GCP_LAT_INTRAZONE_MS if zone_label == "intra-zone" else (GCP_LAT_INTERZONE_MS if zone_label == "inter-zone" else D)
                    if d <= lat_limit and d < INF_V and infra.cpuCap[h] >= cpu_need and infra.ramCap[h] >= ram_need:
                        _, f = infer_link_type(d)
                        e = bw * d * f
                        tier = get_gcp_tier_label(infra.cpuCap[h])
                        cands.append((h, d, e, tier))
                cands.sort(key=lambda x: x[2])
                top = cands[:4]
                h_list = "  ".join(f"H{h}(d={d}ms,E={e:.0f},{t})" for h,d,e,t in top)
                lines.append(f"    Within {zone_label}: {len(cands)} VMs fit — top: {h_list}")
        else:
            lines.append(f"    Both FREE — OPTIMAL: colocate C{src_c}+C{dst_c} on same VM -> E_link=0")
            both_cpu = appli.cpuComp[src_c] + appli.cpuComp[dst_c]
            both_ram = appli.ramComp[src_c] + appli.ramComp[dst_c]
            cands_both = [(h, infra.cpuCap[h], infra.ramCap[h])
                         for h in range(infra.nbH)
                         if infra.cpuCap[h] >= both_cpu and infra.ramCap[h] >= both_ram]
            lines.append(f"    {len(cands_both)} VMs can fit both (CPU>={both_cpu}, RAM>={both_ram})")

        lines.append("")

    # =========================================================================
    # 4. OPTIMAL PLACEMENT STRATEGY
    # =========================================================================
    # =========================================================================
    # 4. OPTIMAL PLACEMENT STRATEGY — consolidation-aware
    # =========================================================================
    lines.append("=== GCP OPTIMAL PLACEMENT STRATEGY ===")
    lines.append("")
    lines.append("ENERGY FORMULA: P_total = P_links + P_nodes  [datastream, continuous power W]")
    lines.append(f"  P_links = sum_l( bw_l × sp_latency(hs_l, ht_l) × factor(type) )  [GCP-aware, [5][6][7]]")
    lines.append(f"  P_nodes = sum_h_active( {GCP_PUE} × [{GCP_P_VCPU_IDLE_W}W×cpu_cap + {GCP_P_VCPU_ACTIVE_W-GCP_P_VCPU_IDLE_W}W×cpu_used] )  [SPECpower [4]]")
    lines.append(f"  COLOCATION: if h_src == h_dst -> E_link = 0  (always prefer colocation)")
    lines.append(f"  SAME-ZONE (≤{GCP_LAT_INTRAZONE_MS}ms): factor={GCP_FACTOR_INTRAZONE} — SAME-REGION (≤{GCP_LAT_INTERZONE_MS}ms): factor={GCP_FACTOR_INTERZONE} — CROSS-REGION (>{GCP_LAT_INTERZONE_MS}ms): factor={GCP_FACTOR_CROSSREGION} ← AVOID")
    lines.append("")

    # Bw total entre DZ et libres, et entre libres eux-mêmes
    dz_hosts = set(dz_map.values())
    bw_dz_to_free = {}   # dz_host -> total bw vers composants libres
    bw_free_internal = 0  # bw total entre composants libres
    for l, (src_c, dst_c) in enumerate(appli.linkPerServ):
        bw = appli.bdwPair[l]
        src_free = src_c not in dz_map
        dst_free = dst_c not in dz_map
        src_dz   = src_c in dz_map
        dst_dz   = dst_c in dz_map
        if src_dz and dst_free:
            dh = dz_map[src_c]
            bw_dz_to_free[dh] = bw_dz_to_free.get(dh, 0) + bw
        elif dst_dz and src_free:
            dh = dz_map[dst_c]
            bw_dz_to_free[dh] = bw_dz_to_free.get(dh, 0) + bw
        elif src_free and dst_free:
            bw_free_internal += bw

    lines.append("TRAFFIC STRUCTURE (critical for placement decision):")
    lines.append(f"  bw DZ<->free components  = {sum(bw_dz_to_free.values())}  (paid once per ms of separation)")
    lines.append(f"  bw free<->free (internal) = {bw_free_internal}  (= 0 if all free components colocated)")
    lines.append(f"  RATIO internal/DZ = {bw_free_internal / max(1, sum(bw_dz_to_free.values())):.1f}x")
    lines.append(f"  => Colocating free components eliminates {bw_free_internal} bw of internal E_links.")
    lines.append(f"  => The residual cost is bw_DZ * sp_lat(DZ_host, chosen_host).")
    lines.append(f"  => GOAL: find host h* that minimizes bw_DZ*d(DZ,h) + E_node(h).")
    lines.append("")

    # =========================================================================
    # 5. RANKED HOST TABLE — trié par vrai coût de consolidation
    # =========================================================================
    lines.append("=== GCP CONSOLIDATION VM RANKING ===")
    lines.append("VMs ranked by TRUE GCP consolidation cost = bw_DZ×d(DZ,h)×factor(type) + E_node(h).")
    lines.append("(Only VMs that can fit ALL free components — CPU and RAM.)")
    lines.append("")
    lines.append(f"  {'VM':>5} {'d_DZ':>6} {'type':>12} {'CPU':>5} {'RAM':>8} {'tier':>10}  E_links  E_node  E_total  [RANK]")

    total_bw_dz = sum(bw_dz_to_free.values())
    consolidation_scores = []
    for h in range(infra.nbH):
        if h in dz_hosts: continue
        if infra.cpuCap[h] < total_cpu: continue
        if infra.ramCap[h] < total_ram: continue
        e_link_consolidation = 0
        for dz_h, bw_dz in bw_dz_to_free.items():
            d = infra.sp_latency[dz_h][h]
            if d >= INF_V: continue
            _, factor = infer_link_type(d)
            e_link_consolidation += bw_dz * d * factor
        if e_link_consolidation == 0 and total_bw_dz > 0:
            continue
        e_node = node_power_w(total_cpu, infra.cpuCap[h])
        e_total = e_link_consolidation + e_node
        tier = get_gcp_tier_label(infra.cpuCap[h])
        d_dz = infra.sp_latency[list(dz_hosts)[0]][h]
        ltype, _ = infer_link_type(d_dz)
        consolidation_scores.append((h, d_dz, infra.cpuCap[h], infra.ramCap[h],
                                     tier, e_link_consolidation, e_node, e_total, ltype))

    consolidation_scores.sort(key=lambda x: x[7])

    for rank, (h, d_dz, cpu, ram, tier, el, en, et, ltype) in enumerate(consolidation_scores[:15], 1):
        marker = " <-- BEST" if rank == 1 else (" <-- 2nd" if rank == 2 else "")
        lines.append(f"  H{h:>4}  {d_dz:>5}ms  {ltype:>12}  {cpu:>5}  {ram:>8}  {tier:>10}  {el:>7.0f}  {en:>6.1f}  {et:>7.1f}{marker}")

    if consolidation_scores:
        best_h = consolidation_scores[0][0]
        best_et = consolidation_scores[0][7]
        lines.append("")
        lines.append(f"  RECOMMENDATION: Place ALL free components on H{best_h}.")
        lines.append(f"  Expected E_total ≈ {best_et:.0f}W  (all internal links colocated -> E=0)")
        lines.append(f"  This is the consolidation optimum for this instance.")
    lines.append("")

    # Hôtes partiels (ne peuvent pas tout accueillir mais peuvent en prendre une partie)
    lines.append("PARTIAL HOSTS (cannot fit all, but useful for 2-host splits):")
    partial = []
    for h in range(infra.nbH):
        if h in dz_hosts: continue
        if infra.cpuCap[h] >= total_cpu: continue  # déjà dans la liste complète
        if infra.cpuCap[h] < max(appli.cpuComp[c] for c in free_comps): continue
        d_dz = min(infra.sp_latency[dh][h] for dh in dz_hosts if infra.sp_latency[dh][h] < INF_V)
        tier = get_gcp_tier_label(infra.cpuCap[h])
        partial.append((h, d_dz, infra.cpuCap[h], infra.ramCap[h], tier))
    partial.sort(key=lambda x: x[1])
    lines.append(f"  {'Host':>5} {'d_DZ':>6} {'CPU':>5} {'RAM':>8} {'tier':>6}")
    for h, d, cpu, ram, tier in partial[:8]:
        lines.append(f"  H{h:>4}  {d:>5}ms  {cpu:>5}  {ram:>8}  {tier}")
    
    # =========================================================================
    # 6. PLACEMENT TEMPLATE
    # =========================================================================
    lines.append("")
    lines.append("=== PLACEMENT TEMPLATE ===")
    lines.append("Your task: Assign each FREE component to a host to minimize total energy.")
    lines.append("")
    lines.append("Fixed placements (DO NOT CHANGE):")
    for ci, hi in appli.DZ:
        lines.append(f"  C{ci} -> H{hi}")
    lines.append("")
    lines.append("Free components to place:")
    for c in free_comps:
        lines.append(f"  C{c}: needs CPU={appli.cpuComp[c]}, RAM={appli.ramComp[c]}")
    lines.append("")
    lines.append("OUTPUT FORMAT: JSON object mapping component ID to host ID")
    lines.append("Example: {" + ", ".join(f'"{c}": <host_id>' for c in range(appli.nbC)) + "}")

    # =========================================================================
    # 7. OBJECTIVE WEIGHTS
    # =========================================================================
    lines.append("")
    lines.append(f"=== OBJECTIVE ===")
    lines.append(f"Weights: w_lat={w[0]}, w_cost={w[1]}, w_energy={w[2]}")
    if w[2] > 0:
        lines.append("ENERGY is the primary objective (datastream: continuous power [W])!")
        lines.append("P_total = P_links + P_nodes  (minimize continuous power draw)")
        lines.append(f"P_links = sum_l( bw_l × sp_latency(hs_l, ht_l) × factor(type) )  [GCP-aware [5][6][7]]")
        lines.append(f"P_nodes = sum_h( {GCP_PUE} × [{GCP_P_VCPU_IDLE_W}W×cpu_cap + {GCP_P_VCPU_ACTIVE_W-GCP_P_VCPU_IDLE_W}W×cpu_used] )  [SPECpower [4]]")
        lines.append(f"  gcp-large  (cpu>=32): P_idle={GCP_PUE*GCP_P_VCPU_IDLE_W*32:.1f}W  P_max={GCP_PUE*GCP_P_VCPU_ACTIVE_W*32:.1f}W  (32vCPU example)")
        lines.append(f"  gcp-medium (cpu=8):   P_idle={GCP_PUE*GCP_P_VCPU_IDLE_W*8:.1f}W   P_max={GCP_PUE*GCP_P_VCPU_ACTIVE_W*8:.1f}W")
        lines.append(f"  gcp-small  (cpu=2):   P_idle={GCP_PUE*GCP_P_VCPU_IDLE_W*2:.1f}W   P_max={GCP_PUE*GCP_P_VCPU_ACTIVE_W*2:.1f}W")

    return "\n".join(lines)
def _build_bottleneck_feedback(sol, infra, appli):
    """Feedback structuré avec analyse des chemins déployés et suggestions d'optimisation."""
    if sol is None:
        return ""
    lines = []
    lines.append(f"\n=== FEEDBACK ON CURRENT SOLUTION (obj={sol.objective_int}) ===")

    if not sol.feasible:
        lines.append(f"STATUS: INFEASIBLE! Violations: {sol.violations}")
        lines.append(f"FIX THESE FIRST before optimizing the objective.")
        return "\n".join(lines)

    lines.append(f"STATUS: FEASIBLE")
    lines.append(f"  P_total={sol.p_total_w:.2f}W (P_link={sol.p_link_w:.2f}, P_node={sol.p_node_w:.2f}) [datastream]")
    lines.append(f"  Total Latency: {sol.total_latency}ms")
    lines.append(f"  Active hosts: {len(sol.active_hosts)} -> {sorted(sol.active_hosts)}")
    lines.append(f"  Placement: {sol.placement}")
    
    # =========================================================================
    # LINK-BY-LINK PATH ANALYSIS
    # =========================================================================
    lines.append(f"\n  DEPLOYED LINK PATHS (sorted by energy cost):")
    
    link_analysis = []
    for l in range(appli.nbL):
        sC, tC = appli.linkPerServ[l]
        bw = appli.bdwPair[l]
        max_lat = appli.latPair[l]
        hs, ht = sol.placement[sC], sol.placement[tC]
        
        if hs == ht:
            # Colocalisés - énergie = 0
            link_analysis.append({
                'link': l, 'src_c': sC, 'dst_c': tC, 'bw': bw,
                'src_h': hs, 'dst_h': ht,
                'path': [hs], 'hops': 0, 'latency': 0, 'energy': 0,
                'colocated': True
            })
        else:
            path = infra.sp_path.get((hs, ht), [hs, ht])
            latency = infra.sp_latency[hs][ht] if infra.sp_latency[hs][ht] < INF_V else 9999
            # Calcul GCP-aware : E_link(l) = bw × sp_lat × factor(type) [5][6][7]
            ltype, factor = infer_link_type(latency)
            energy = 0
            for k in range(len(path) - 1):
                arc_lat = infra.arc_lat.get((path[k], path[k + 1]), 0)
                _, arc_factor = infer_link_type(arc_lat)
                energy += bw * arc_lat * arc_factor
            
            link_analysis.append({
                'link': l, 'src_c': sC, 'dst_c': tC, 'bw': bw,
                'src_h': hs, 'dst_h': ht,
                'path': path, 'hops': len(path) - 1, 'latency': latency, 'energy': energy,
                'colocated': False
            })
    
    # Trier par énergie décroissante
    link_analysis.sort(key=lambda x: -x['energy'])
    
    total_link_energy = sum(la['energy'] for la in link_analysis)
    
    for la in link_analysis:
        if la['colocated']:
            lines.append(f"    L{la['link']}: C{la['src_c']}->C{la['dst_c']} on H{la['src_h']} "
                        f"[COLOCATED] -> E=0, lat=0ms ✓")
        else:
            path_str = "->".join(f"H{p}" for p in la['path'])
            pct = 100 * la['energy'] / max(1, total_link_energy)
            ltype, factor = infer_link_type(la['latency'])
            lines.append(f"    L{la['link']}: C{la['src_c']}->C{la['dst_c']} | "
                        f"Path: {path_str} ({la['hops']} hops) | "
                        f"lat={la['latency']}ms [{ltype},×{factor}] | E={la['energy']:.0f} ({pct:.0f}%)")
    
    # =========================================================================
    # OPTIMIZATION OPPORTUNITIES
    # =========================================================================
    lines.append(f"\n  OPTIMIZATION OPPORTUNITIES:")
    
    # Trouver les liens non colocalisés avec le plus d'énergie
    non_colocated = [la for la in link_analysis if not la['colocated']]
    
    if non_colocated:
        worst = non_colocated[0]
        wltype, wfactor = infer_link_type(worst['latency'])
        lines.append(f"    1. WORST LINK: L{worst['link']} (C{worst['src_c']}->C{worst['dst_c']}) "
                    f"costs {worst['energy']:.0f} energy [{wltype},×{wfactor}] ({worst['hops']} hops)")
        if wltype == "cross-region":
            savings_if_intra = worst['bw'] * worst['latency'] * (GCP_FACTOR_CROSSREGION - GCP_FACTOR_INTRAZONE)
            lines.append(f"       >> CROSS-REGION penalty: factor={GCP_FACTOR_CROSSREGION} × {worst['latency']}ms")
            lines.append(f"       >> PRIORITY: Move to same zone (factor=1.0) — would save ~{savings_if_intra:.0f} energy")

        src_c, dst_c = worst['src_c'], worst['dst_c']
        src_h, dst_h = worst['src_h'], worst['dst_h']
        cpu_on_src = sum(appli.cpuComp[c] for c in range(appli.nbC) if sol.placement[c] == src_h)
        cpu_on_dst = sum(appli.cpuComp[c] for c in range(appli.nbC) if sol.placement[c] == dst_h)
        can_move_dst_to_src = (cpu_on_src + appli.cpuComp[dst_c] <= infra.cpuCap[src_h])
        can_move_src_to_dst = (cpu_on_dst + appli.cpuComp[src_c] <= infra.cpuCap[dst_h])

        if can_move_dst_to_src:
            lines.append(f"       -> Colocate: Move C{dst_c} to H{src_h} (saves {worst['energy']:.0f} energy)")
        if can_move_src_to_dst:
            lines.append(f"       -> Colocate: Move C{src_c} to H{dst_h} (saves {worst['energy']:.0f} energy)")

        if len(non_colocated) > 1:
            total_potential_savings = sum(la['energy'] for la in non_colocated)
            lines.append(f"    2. TOTAL POTENTIAL SAVINGS: {total_potential_savings:.0f} "
                        f"if all {len(non_colocated)} non-colocated links are colocated")
    else:
        lines.append(f"    All links are colocated - E_links is already optimal (0)!")

    lines.append(f"\n  GCP VM UTILIZATION:")
    for h in sorted(sol.active_hosts):
        comps_on_h = [c for c in range(appli.nbC) if sol.placement[c] == h]
        cpu_used = sum(appli.cpuComp[c] for c in comps_on_h)
        cpu_free = infra.cpuCap[h] - cpu_used
        p_w = node_power_w(cpu_used, infra.cpuCap[h])
        tier_label = get_gcp_tier_label(infra.cpuCap[h])
        lines.append(f"    H{h} ({tier_label}): {comps_on_h} | CPU: {cpu_used}/{infra.cpuCap[h]} (free={cpu_free}) | E_node={p_w:.1f}W")

    if sol.p_link_w > sol.p_node_w * 2:
        lines.append(f"\n  >>> P_link ({sol.p_link_w:.1f}) >> P_node ({sol.p_node_w:.1f})")
        lines.append(f"  >>> GCP PRIORITY: Minimize cross-region traffic! Colocate or move to same zone.")
    
    return "\n".join(lines)


def _generate_seed_placements(infra, appli, bounds, w):
    """A4: Genere des placements seed diversifies pour guider le LLM."""
    seeds = []
    dz = {ci: hi for ci, hi in appli.DZ}
    free = [c for c in range(appli.nbC) if c not in dz]
    free_cpu = sum(appli.cpuComp[c] for c in free)
    free_ram = sum(appli.ramComp[c] for c in free)

    # Seed 1: Greedy Energy FFD
    seeds.append(("FFD greedy energy baseline", greedy_energy_placement(infra, appli)))

    # Seed 2: Consolidate all free components on best GCP VM near DZ
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

    # Seed 5: Zone-consolidation — place all free components on the nearest intra-zone VM to DZ
    dz_h_main_s5 = list(dz.values())[0] if dz else 0
    intrazone_hosts = [h for h in range(infra.nbH)
                       if h not in dz.values()
                       and infra.sp_latency[dz_h_main_s5][h] <= GCP_LAT_INTRAZONE_MS
                       and infra.sp_latency[dz_h_main_s5][h] < INF_V]
    if intrazone_hosts:
        rh = min(intrazone_hosts, key=lambda h: (infra.sp_latency[dz_h_main_s5][h], -infra.cpuCap[h]))
        pl = dict(dz); cu = [0] * infra.nbH; ru = [0] * infra.nbH
        for ci, hi in dz.items():
            cu[hi] += appli.cpuComp[ci]; ru[hi] += appli.ramComp[ci]
        for c in free:
            if (cu[rh] + appli.cpuComp[c] <= infra.cpuCap[rh] and
                    ru[rh] + appli.ramComp[c] <= infra.ramCap[rh]):
                pl[c] = rh; cu[rh] += appli.cpuComp[c]; ru[rh] += appli.ramComp[c]
            else:
                for h2 in sorted(range(infra.nbH),
                                  key=lambda h: infra.sp_latency[rh][h]
                                  if infra.sp_latency[rh][h] < INF_V else 9999):
                    if (cu[h2] + appli.cpuComp[c] <= infra.cpuCap[h2] and
                            ru[h2] + appli.ramComp[c] <= infra.ramCap[h2]):
                        pl[c] = h2; cu[h2] += appli.cpuComp[c]; ru[h2] += appli.ramComp[c]
                        break
                else:
                    pl[c] = 0
        seeds.append((f"GCP intra-zone H{rh}", pl))

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
                        sc += bw * d  # Minimize link energy (linear proxy [5][6])
                # Prefer already-active hosts (no extra node cost)
                if cu[h] == 0 and h not in set(dz.values()):
                    # Penalty for activating a new host (P_eff_idle, tier-aware [1][2])
                    p_idle, _, pue, _ = get_tier_params(infra.cpuCap[h])
                    sc += p_idle * pue
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
                     f"lat={best.total_latency}ms P_total={best.p_total_w:.2f}W ---")
        parts.append(f"Placement: {best.placement}")
    
    # Diameter reminder compact — ancre le raisonnement à chaque appel
    D = infra.diameter
    parts.append(
        f"\nGCP NETWORK REMINDER: E_link=bw×sp_lat×factor. "
        f"factor: ≤{GCP_LAT_INTRAZONE_MS}ms→{GCP_FACTOR_INTRAZONE}(intra-zone), "
        f"≤{GCP_LAT_INTERZONE_MS}ms→{GCP_FACTOR_INTERZONE}(inter-zone), "
        f">{GCP_LAT_INTERZONE_MS}ms→{GCP_FACTOR_CROSSREGION}(cross-region!). "
        f"D={D}ms. coloc→E=0. AVOID cross-region (×{GCP_FACTOR_CROSSREGION} penalty)."
    )

    # Strategy hint
    if strategy_hint:
        parts.append(f"\nHINT[{iteration}/{max_iter}]: {strategy_hint}")
    
    parts.append(f'\nRespond JSON: {{"placement": {{"0": host_id, ...}}, "reasoning": "..."}}')
    
    return "\n".join(parts)


def run_llm(infra, appli, bounds, w, provider, model, max_iter=5, verbose=True, use_seeds=False):
    """OPRO-inspired LLM optimizer with optimization trajectory and multi-solution generation.
    By default, the LLM starts from scratch without heuristic seed placements (use_seeds=False)."""
    
    # Vérification préliminaire: infrastructure trop grande pour LLM?
    if infra.nbH > 200:
        if verbose:
            print(f"    AVERTISSEMENT: Infrastructure tres grande ({infra.nbH} hotes).")
            print(f"    Le prompt pourrait etre trop long pour certains LLMs.")
            print(f"    Utilisation d'une description resumee...")
    
    total_tok_in = 0; total_tok_out = 0; total_llm_time = 0.0
    best = None; t0 = time.time()
    
    # OPRO trajectory: list of (score, placement_dict, reasoning_str)
    trajectory = []

    # --- Phase 1: Seed evaluation (A2+A4+A7+A8) ---
    if use_seeds:
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
                          f"lat={sol.total_latency}ms P_total={sol.p_total_w:.2f}W *BEST*")
            elif verbose:
                tag = "OK" if sol.feasible else "FAIL"
                print(f"      {name}: obj={sol.objective_int} [{tag}]")
    else:
        if verbose:
            print("    Mode sans seeds: le LLM part de zero (exploration pure)")

    # --- Phase 2: Build problem description (avec limite pour grandes infras) ---
    max_hosts_detail = 50 if infra.nbH <= 200 else 30
    problem_desc = _build_problem_description(infra, appli, bounds, w, max_hosts_detail)

    # --- Phase 3: OPRO-style iterative optimization ---
    # OPRO key parameters (from paper Section 5.3):
    # - 8 solutions per step (we use multi-call)
    # - Temperature 1.0 default (we adapt: low→high)
    # - Top 20 solutions in trajectory
    # - 3 exemplars
    # - Ascending order (worst→best) for recency bias
    
    # Consolidation-aware strategies — fondées sur le vrai critère de coût
    # Le tableau CONSOLIDATION HOST RANKING dans le prompt donne le meilleur hôte.
    D = infra.diameter
    dz_map = {ci: hi for ci, hi in appli.DZ}  # reconstruit localement pour les stratégies
    total_cpu = sum(appli.cpuComp[c] for c in range(appli.nbC) if c not in dz_map)
    total_ram = sum(appli.ramComp[c] for c in range(appli.nbC) if c not in dz_map)

    # Identifier le meilleur hôte de consolidation (top du ranking)
    dz_hosts_set = set(dz_map.values())
    dz_h_main = list(dz_map.values())[0]   # DZ principal (anchor pour les distances)
    bw_dz_free = {}
    bw_free_int = 0
    for l2, (sc2, tc2) in enumerate(appli.linkPerServ):
        bw2 = appli.bdwPair[l2]
        if sc2 in dz_map and tc2 not in dz_map:
            dh2 = dz_map[sc2]; bw_dz_free[dh2] = bw_dz_free.get(dh2, 0) + bw2
        elif tc2 in dz_map and sc2 not in dz_map:
            dh2 = dz_map[tc2]; bw_dz_free[dh2] = bw_dz_free.get(dh2, 0) + bw2
        elif sc2 not in dz_map and tc2 not in dz_map:
            bw_free_int += bw2
    total_bw_dz2 = sum(bw_dz_free.values())

    # Calculer ranking consolidation
    # ── Logique GCP-aware : trier par ZONE d'abord, puis par coût réel (el+en).
    #
    # Principe de la stratégie It1/S1 :
    #   On cherche le meilleur nœud DANS LA MÊME ZONE QUE LE DZ en premier,
    #   puis inter-zone, puis cross-région — jamais l'inverse.
    #
    #   Au sein de chaque zone, on trie par el+en (coût énergétique réel) :
    #   cela permet de choisir un nœud frugal (petit cpu_cap) si le réseau
    #   est peu chargé, ou un nœud proche si bw est élevé.
    #
    # Zones GCP (en ordre de priorité) :
    #   0 = DZ lui-même        (d=0,  E_link=0)              ← essayé en 1er
    #   1 = intra-zone         (d≤2ms,  factor=1.0)          ← voisins directs
    #   2 = inter-zone         (d≤25ms, factor=1.5)          ← même région
    #   3 = cross-région       (d>25ms, factor=2.0)          ← à éviter
    consol_rank = []
    for hh in range(infra.nbH):
        if hh in dz_hosts_set: continue
        if infra.cpuCap[hh] < total_cpu: continue
        if infra.ramCap[hh] < total_ram: continue
        d_to_dz = infra.sp_latency[dz_h_main][hh] if dz_h_main is not None else INF_V
        if d_to_dz >= INF_V: continue
        # Tier de zone : 0=DZ, 1=intra, 2=inter, 3=cross
        if d_to_dz == 0:   zone_tier = 0
        elif d_to_dz <= GCP_LAT_INTRAZONE_MS:  zone_tier = 1
        elif d_to_dz <= GCP_LAT_INTERZONE_MS:  zone_tier = 2
        else:                                   zone_tier = 3
        # Énergie liens DZ→hh
        el = 0
        for dh2, bw2 in bw_dz_free.items():
            d = infra.sp_latency[dh2][hh]
            if d < INF_V:
                _, factor = infer_link_type(d)
                el += bw2 * d * factor
        # Énergie nœud (P_idle proportionnelle au cpu_cap — préférer le plus petit suffisant)
        en = node_power_w(total_cpu, infra.cpuCap[hh])
        # Clé : (zone_tier, el+en) — zone prime sur tout, puis coût réel dans la zone
        consol_rank.append((zone_tier, el + en, hh, el, en))
    consol_rank.sort()
    # Réordonner pour compatibilité avec le code en aval : (score_total, hh, el, en)
    consol_rank = [(row[1], row[2], row[3], row[4]) for row in consol_rank]

    best_consol_h   = consol_rank[0][1]  if consol_rank else None
    best_consol_e   = consol_rank[0][0]  if consol_rank else 0
    second_consol_h = consol_rank[1][1]  if len(consol_rank) > 1 else best_consol_h
    third_consol_h  = consol_rank[2][1]  if len(consol_rank) > 2 else second_consol_h
    d_best = infra.sp_latency[dz_h_main][best_consol_h] if best_consol_h is not None else 0
    d_second = infra.sp_latency[dz_h_main][second_consol_h] if second_consol_h is not None else 0

    # Pré-calcul de la solution consolidée optimale pour S1 — injectée directement
    consol_placement_str = "{" + ", ".join(
        f'"{c}": "H{best_consol_h}"' for c in range(appli.nbC) if c not in dz_map
    ) + "}"
    # Ajouter les DZ
    dz_str = ", ".join(f'"{c}": "H{h}"' for c, h in dz_map.items())
    full_consol_str = "{" + dz_str + (", " if dz_str else "") + ", ".join(
        f'"{c}": "H{best_consol_h}"' for c in range(appli.nbC) if c not in dz_map
    ) + "}"

    opro_strategies = [
        # S1 — GCP-ZONE-CONSOLIDATE : consolidation sur le nœud le plus PROCHE du DZ
        #   Logique : DZ d'abord (d=0), puis voisins intra-zone (d≤2ms), puis inter-zone.
        #   On NE cherche PAS le plus gros nœud (qui serait loin et gaspillerait P_idle).
        #   On cherche le plus proche qui peut JUSTE accueillir tous les composants libres.
        f"GCP-ZONE-CONSOLIDATE: Place ALL *free* components on H{best_consol_h} "
        f"(closest VM to DZ H{dz_h_main} that fits all free components — "
        f"d={infra.sp_latency[dz_h_main][best_consol_h] if best_consol_h is not None else 0}ms from DZ). "
        f"Priority: DZ itself (d=0) > intra-zone (d≤2ms) > inter-zone (d≤25ms) > cross-region. "
        f"VERIFIED: sum(cpu_free)={total_cpu} <= cpuCap[H{best_consol_h}]="
        f"{infra.cpuCap[best_consol_h] if best_consol_h is not None else '?'}, "
        f"sum(ram_free)={total_ram} <= ramCap[H{best_consol_h}]="
        f"{infra.ramCap[best_consol_h] if best_consol_h is not None else '?'}. "
        f"ALL internal free-free links → E_link=0 (collocated). "
        f"Expected E_total ≈ {best_consol_e:.0f}W. "
        f"Output EXACTLY this JSON: {full_consol_str}",

        # S2 — GCP-RANK2 : rang 2 du ranking GCP
        f"GCP-RANK2: Place ALL free components on H{second_consol_h} (rank #2 in GCP ranking). "
        f"VERIFIED: sum(cpu_free)={total_cpu} <= cpuCap[H{second_consol_h}]="
        f"{infra.cpuCap[second_consol_h] if second_consol_h is not None else '?'}. "
        f"DZ component c{list(dz_map.keys())[0]} stays on H{dz_h_main} — do NOT move it. "
        f"GCP cost = bw_DZ({total_bw_dz2}) × d({d_second}ms) × factor + E_node. "
        f"Output JSON with every free comp mapped to H{second_consol_h}.",

        # S3 — GCP-FIX-CROSSREGION : priorité sur liens cross-région
        f"GCP-FIX-CROSSREGION: Look at FEEDBACK 'DEPLOYED LINK PATHS'. "
        f"Find links with [cross-region,×{GCP_FACTOR_CROSSREGION}] — these dominate the energy cost. "
        f"For the worst cross-region link: move BOTH endpoints to a VM in the same zone as H{dz_h_main} "
        f"(sp_latency ≤ {GCP_LAT_INTRAZONE_MS}ms → factor={GCP_FACTOR_INTRAZONE}). "
        f"Check CPU/RAM before moving. DZ c{list(dz_map.keys())[0]} cannot move from H{dz_h_main}. "
        f"Cross-region at 85ms costs 85×2.0=170× more than intra-zone at 1ms.",

        # S4 — GCP-ZONE-SPLIT : split 2 VMs dans la MÊME ZONE
        f"GCP-ZONE-SPLIT: If no single VM fits all free components, split into 2 groups "
        f"BOTH in the same zone as H{dz_h_main} (sp_latency ≤ {GCP_LAT_INTRAZONE_MS}ms, factor={GCP_FACTOR_INTRAZONE}). "
        f"Group A: components with highest bw to DZ → place on VM nearest to H{dz_h_main} in same zone. "
        f"Group B: remaining → place on another same-zone VM. "
        f"NEVER use cross-region VMs (>25ms) for this split. "
        f"Minimize: bw_A×d_A×1.0 + bw_B×d_B×1.0 + bw_AB×d_AB×1.0.",

        # S5 — GCP-MICRO-OPT : éliminer cross-région d'abord, puis inter-zone
        f"GCP-MICRO-OPT: Take the BEST feasible solution from trajectory. "
        f"Step 1: For each component NOT in the same zone as H{dz_h_main}: "
        f"check if moving it to H{best_consol_h} reduces E_total AND respects CPU/RAM. "
        f"Step 2: Eliminate cross-region links (factor={GCP_FACTOR_CROSSREGION}) first — highest penalty. "
        f"Step 3: Then tackle inter-zone links (factor={GCP_FACTOR_INTERZONE}). "
        f"DZ c{list(dz_map.keys())[0]} is locked on H{dz_h_main}.",

        # S6 — GCP-RANK3 : rang 3
        f"GCP-RANK3: Place ALL free components on H{third_consol_h} (rank #3 in GCP CONSOLIDATION VM RANKING). "
        f"DZ c{list(dz_map.keys())[0]} stays on H{dz_h_main}. "
        f"Output JSON with every free comp on H{third_consol_h}.",

        # S7 — GCP-FEASIBILITY : faisabilité d'abord
        f"GCP-FEASIBILITY: If previous attempts were INFEASIBLE, focus on C2 (CPU/RAM). "
        f"H{best_consol_h} (CPU={infra.cpuCap[best_consol_h] if best_consol_h is not None else '?'}) "
        f"can fit total_cpu={total_cpu}, total_ram={total_ram}. "
        f"DZ: c{list(dz_map.keys())[0]} MUST stay on H{dz_h_main}. "
        f"Output: every free comp on H{best_consol_h}. GCP PUE=1.10 uniform — no tier penalty.",

        # S8 — GCP-VERIFY-FINAL : vérification finale sans cross-région
        f"GCP-VERIFY-FINAL: Take best feasible solution. "
        f"Is every free component in the same zone as H{dz_h_main} (sp_lat ≤ {GCP_LAT_INTRAZONE_MS}ms)? "
        f"For each non-intra-zone component: compute savings if moved to H{best_consol_h}. "
        f"Are there cross-region links (factor={GCP_FACTOR_CROSSREGION})? → MUST fix. "
        f"Is E_total < {best_consol_e*1.1:.0f}W? If not, force full consolidation on H{best_consol_h}.",
    ]

    # OPRO: generate solutions per step — fewer in exploit phases, more in explore
    effective_steps = max(1, max_iter)
    stale_iterations = 0  # Track iterations without improvement for early convergence
    max_stale = 5  # FIX v9: augmenté de 3→5 pour éviter convergence prématurée
    
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
        
        # FIX v9: 2 solutions par itération systématiquement (diversité accrue)
        solutions_per_step = 2

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
        
        # Track convergence — stale seulement si au moins 1 solution faisable produite cette iter
        # FIX v9: les itérations 100% infaisables ne comptent plus comme stale
        has_feasible = best is not None and best.feasible
        has_feasible_this_step = any(s.feasible for s, _ in step_solutions)
        if step_improved:
            stale_iterations = 0
        elif has_feasible and has_feasible_this_step:
            stale_iterations += 1
        # else: aucune solution faisable cette iter — ne pas pénaliser

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

    # --- Phase 4: Post-LLM local search multi-start renforcée ---
    # FIX v9: max_iter 400→800, 5 départs, + injection de la solution consolidée optimale
    if best and best.feasible:
        feasible_starts = []
        seen_pl = set()
        for score, pl_dict, _ in sorted(trajectory, key=lambda x: x[0]):
            pl_key = tuple(sorted(pl_dict.items()))
            if pl_key not in seen_pl:
                test = evaluate_solution(pl_dict, infra, appli, bounds, *w)
                if test.feasible:
                    feasible_starts.append(pl_dict)
                    seen_pl.add(pl_key)
                    if len(feasible_starts) >= 4:
                        break
        if not feasible_starts:
            feasible_starts = [best.placement]
        elif dict(feasible_starts[0]) != dict(best.placement):
            feasible_starts.insert(0, best.placement)

        # Injecter la solution consolidée optimale comme départ synthétique supplémentaire
        if best_consol_h is not None:
            synth_pl = dict(dz_map)
            for c in range(appli.nbC):
                if c not in dz_map:
                    synth_pl[c] = best_consol_h
            synth_sol = evaluate_solution(synth_pl, infra, appli, bounds, *w)
            if synth_sol.feasible:
                synth_key = tuple(sorted(synth_pl.items()))
                if synth_key not in seen_pl:
                    feasible_starts.append(synth_pl)
                    if verbose:
                        print(f"    [v9] Départ synthétique consolidé H{best_consol_h}: "
                              f"P_total={synth_sol.p_total_w:.2f}W (injecté)")
                    if synth_sol.objective < best.objective:
                        best = synth_sol

        for start_pl in feasible_starts[:5]:
            post_best = standalone_local_search(start_pl, infra, appli, bounds, w, max_iter=800)
            if post_best.feasible and post_best.objective < best.objective:
                best = post_best
        if verbose:
            print(f"    Post-LS: obj={best.objective_int} P_total={best.p_total_w:.2f}W")

    t_total = time.time() - t0
    if best is None:
        best = evaluate_solution(greedy_placement(infra, appli), infra, appli, bounds, *w)
    best.solve_time = t_total

    # E_solver LLM — modèle tokens [8][9] avec incertitude
    total_tok = total_tok_in + total_tok_out
    cost_in, cost_out = COST_PER_1K_TOKENS.get(provider, (0, 0))
    if provider == "ollama":
        # Mesure directe P*T si disponible [8], sinon fallback tokens
        e_solver, e_min, e_max = llm_energy_wh(
            provider, total_tok_in, total_tok_out,
            p_gpu_w=P_CPU_SOLVER_W, t_inference_s=total_llm_time)
        e_method = f"P*T ({P_CPU_SOLVER_W:.0f}W) local [8]"
    else:
        # API distante : modèle basé tokens [8][9]
        e_solver, e_min, e_max = llm_energy_wh(provider, total_tok_in, total_tok_out)
        eps_in, eps_out, _ = LLM_ENERGY_PROFILES.get(provider, (0.005, 0.015, 5.0))
        e_method = f"tok*(eps_in={eps_in},eps_out={eps_out})Wh/1k [8][9]"
    cost_usd = total_tok_in / 1000.0 * cost_in + total_tok_out / 1000.0 * cost_out

    if verbose:
        print(f"  [LLM] Tokens: {total_tok_in}in + {total_tok_out}out = {total_tok}")
        print(f"  [LLM] E_solver={e_solver:.6f}Wh [min={e_min:.6f}, max={e_max:.6f}] | ${cost_usd:.4f}")
        print(f"  [LLM] Note: incertitude x3 sur énergie LLM (variabilité datacenter) [9]")

    return SolverResult(
        name=f"LLM ({model})", sol=best,
        t_total_s=t_total, tokens_in=total_tok_in, tokens_out=total_tok_out,
        e_solver_wh=e_solver, e_solver_wh_min=e_min, e_solver_wh_max=e_max,
        cost_usd=cost_usd, e_method=e_method)


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
    row("P_link (W, proxy norm.)",  lambda r: f"{r.sol.p_link_w:.2f}")
    row("P_node (W, Barroso [1])",   lambda r: f"{r.sol.p_node_w:.2f}")
    row("P_total (W, objectif)",     lambda r: f"{r.sol.p_total_w:.2f}")
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

    # --- ROI ÉNERGÉTIQUE — régime datastream (application permanente) ---
    # Baseline = FFD (premier résultat Glouton)
    ffd = next((r for r in valid if "Glouton" in r.name or "FFD" in r.name or "Greedy" in r.name), valid[-1])
    p_ffd = ffd.sol.p_total_w  # puissance de référence [W ou unités normalisées]

    print(f" {'ROI — DATASTREAM (puissance continue)':^30}" + f"  {'':>{W}} " * N)
    print(f" {'[T*=E_solver[Wh]/ΔP[W]*3600, homogène en s]':^30}" + f"  {'':>{W}} " * N)

    # ΔP : gain de puissance vs baseline Greedy [W normalisés]
    row("ΔP vs Greedy (W norm.)",
        lambda r: f"{p_ffd - r.sol.p_total_w:+.2f}" if r.sol.feasible else "N/A")

    # T* [s] = E_solver[Wh] / ΔP[W] × 3600
    # Homogène si ΔP est en W réels ; ici en unités proxy normalisées
    # → T* indicatif, mais physiquement cohérent en ordre de grandeur [1][2][8][9]
    def tstar_str(r):
        if not r.sol.feasible: return "N/A"
        if "Glouton" in r.name or "FFD" in r.name or "Greedy" in r.name:
            return "baseline"
        delta_p = p_ffd - r.sol.p_total_w
        if delta_p <= 0: return ">∞ (pas de gain)"
        if r.e_solver_wh <= 1e-10: return "~0s"
        tstar_s = r.e_solver_wh / (delta_p / 3600.0)  # Wh/(W·h⁻¹) = h → ×3600 = s... 
        # Correction : T*[s] = E_solver[Wh] × 3600[s/h] / ΔP[W]
        tstar_s = r.e_solver_wh * 3600.0 / delta_p
        return f"{tstar_s:.1f} s (indicatif)"
    row("T* payback [s] (datastream) [9]", tstar_str)

    # Incertitude LLM [9]
    def solver_ci(r):
        if r.e_solver_wh_min <= 0 and r.e_solver_wh_max <= 0: return "-"
        return f"[{r.e_solver_wh_min:.6f}, {r.e_solver_wh_max:.6f}]"
    row("E_solver IC [min,max] Wh [9]",  solver_ci)

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
                # Puissance continue [W] — régime datastream [1][2][5]
                "p_node_w": round(r.sol.p_node_w, 4),
                "p_link_w": round(r.sol.p_link_w, 4),
                "p_total_w": round(r.sol.p_total_w, 4),
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
# AFFICHAGE DU MODÈLE ÉNERGÉTIQUE GCP (rapport / section 3 papier)
# Réf: [2][4][5][6][7]
# =====================================================================
def _print_energy_model(infra):
    """Affiche le résumé du modèle énergétique GCP appliqué à l'infrastructure."""
    from collections import Counter
    tier_counts = Counter()
    for cpu in infra.cpuCap:
        label = get_gcp_tier_label(cpu)
        tier_counts[label] += 1

    print("\n" + "=" * 65)
    print(" MODÈLE ÉNERGÉTIQUE — GCP Cloud Pur (v10-gcp)")
    print(" Réf: Google Env. Report 2023 [2], SPECpower_ssj2008 [4]")
    print("=" * 65)
    print(f"  PUE uniforme GCP : {GCP_PUE}  (tous datacenters Google [2])")
    print(f"  Calibration vCPU : P_idle={GCP_P_VCPU_IDLE_W}W/vCPU, P_active={GCP_P_VCPU_ACTIVE_W}W/vCPU  [4]")
    print(f"\n  Distribution des VMs (inférée depuis cpu_cap) :")
    for label in ["gcp-large", "gcp-medium", "gcp-small"]:
        if tier_counts[label]:
            print(f"    {label:12s}: {tier_counts[label]:4d} VMs")
    print(f"\n  P(h) = {GCP_PUE} × [{GCP_P_VCPU_IDLE_W}W×cpu_cap + {GCP_P_VCPU_ACTIVE_W-GCP_P_VCPU_IDLE_W}W×cpu_used]  [2][4]")
    # Exemples pour quelques tailles de VM
    for cpu in [2, 4, 8, 16, 32]:
        p_idle = node_power_w(0, cpu)  # returns 0 for cpu_used=0
        p_full = node_power_w(cpu, cpu)
        print(f"    {cpu:2d} vCPU: P_idle={GCP_PUE*GCP_P_VCPU_IDLE_W*cpu:.1f}W  P_max={p_full:.1f}W")
    print(f"\n  E_link = bw × latency × factor(type)  [5][6][7]")
    print(f"    intra-zone  (≤{GCP_LAT_INTRAZONE_MS}ms)  : ×{GCP_FACTOR_INTRAZONE}")
    print(f"    inter-zone  (≤{GCP_LAT_INTERZONE_MS}ms) : ×{GCP_FACTOR_INTERZONE}")
    print(f"    cross-region (>{GCP_LAT_INTERZONE_MS}ms) : ×{GCP_FACTOR_CROSSREGION}  ← ÉVITER")
    print(f"  E_solver(CSP/Greedy) = {P_CPU_SOLVER_W}W × t_solve")
    print(f"  ROI T*[s] = E_solver[Wh]×3600 / ΔP[W]  (datastream, homogène)")
    print(f"  E_solver(LLM) = tok_in×eps_in + tok_out×eps_out  [8][9]")
    print(f"  Incertitude LLM : facteur ×3 (variabilité datacenter) [9]")
    print("=" * 65)

# =====================================================================
# MAIN
# =====================================================================
def main():
    pa = argparse.ArgumentParser(description="TPDS2 Benchmark Unifie: CSP / LLM / Glouton (FFD) — Objectif Energie")
    #pa.add_argument("--infra", default="resources/Infra_8nodes_2.properties")
    #pa.add_argument("--appli", default="resources/Appli_4comps_2.properties")
    #pa.add_argument("--infra", default="resources/Infra_16nodes_fog3tier.properties")
    #pa.add_argument("--appli", default="resources/Appli_8comps_smartbuilding.properties")
    #pa.add_argument("--infra", default="resources/Infra_24nodes_dcns.properties")
    #pa.add_argument("--appli", default="resources/Appli_10comps_dcns.properties")
    #pa.add_argument("--infra", default="resources/Infra_28nodes_smartcity.properties")
    #pa.add_argument("--appli", default="resources/Appli_11comps_smartcity.properties")
    #pa.add_argument("--infra", default="resources/Infra_32nodes_hospital.properties")
    #pa.add_argument("--appli", default="resources/Appli_12comps_ehealth.properties")
    pa.add_argument("--infra", default="resources/files43/Infra_1000nodes_city.properties")
    pa.add_argument("--appli", default="resources/files43/Appli_8comps_dense.properties")
    
    pa.add_argument("--w-lat", type=float, default=0.)
    pa.add_argument("--w-cost", type=float, default=0.)
    pa.add_argument("--w-energy", type=float, default=1)
    pa.add_argument("--llm-iter", type=int, default=8)
    # --t-app supprimé : régime datastream, application permanente.
    # Le ROI se calcule via T*[s] = E_solver[Wh]*3600 / ΔP[W], sans paramètre durée.
    pa.add_argument("--n-runs", type=int, default=1,
                    help="Répétitions LLM pour variance (publiable Compas) (défaut: 1)")
    
    # Options de contrôle des solveurs
    pa.add_argument("--skip-csp", action="store_true", help="Ne pas lancer CP-SAT")
    pa.add_argument("--skip-llm", action="store_true", help="Ne pas lancer le LLM")
    pa.add_argument("--skip-greedy", action="store_true", help="Ne pas lancer le Greedy")
    pa.add_argument("--only-csp", action="store_true", help="Lancer UNIQUEMENT le CSP (skip greedy et LLM)")
    pa.add_argument("--only-greedy", action="store_true", help="Lancer UNIQUEMENT le Greedy (skip CSP et LLM)")
    pa.add_argument("--only-llm", action="store_true", help="Lancer UNIQUEMENT le LLM (skip greedy et CSP)")
    
    pa.add_argument("--provider", default=None, help="LLM provider: anthropic/openai/gemini/ollama")
    pa.add_argument("--model", default=None, help="Nom du modele LLM")
    pa.add_argument("--greedy-mode", choices=["energy", "legacy"], default="energy",
                    help="Greedy objective: 'energy' (min E_nodes+E_links) or 'legacy' (cost+latency)")
    pa.add_argument("--use-seeds", action="store_true",
                    help="LLM starts WITH heuristic seed placements (default: sans seeds)")
    args = pa.parse_args()
    
    # Gérer les options --only-*
    if args.only_csp:
        args.skip_greedy = True
        args.skip_llm = True
    elif args.only_greedy:
        args.skip_csp = True
        args.skip_llm = True
    elif args.only_llm:
        args.skip_csp = True
        args.skip_greedy = True
    
    w = (args.w_lat, args.w_cost, args.w_energy)

    print("=" * 80)
    print(" TPDS2 - BENCHMARK UNIFIE (OBJECTIF ENERGIE)")
    print(" CSP (CP-SAT) | LLM | Glouton (FFD)")
    print("=" * 80)

    # --- Chargement ---
    print("\n[1/4] Chargement des donnees...")
    infra = load_infra(args.infra)
    appli = load_appli(args.appli)

    # Validation critique : DZ compatibles avec l'infra chargée
    try:
        validate_dz_against_infra(appli, infra)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Calculer et afficher les metriques du graphe (dont le diametre)
    computed_diameter = print_graph_metrics(infra)
    bounds = compute_norm_bounds(infra, appli)
    print(f"  Infra: {infra.nbH} hotes | Diametre: {infra.diameter}")
    # Afficher le modèle énergétique appliqué [1][2][7]
    _print_energy_model(infra)
    print(f"  Appli: {appli.nbC} composants | {appli.nbL} liens | {appli.nbDZ} DZ")
    print(f"  Poids: w_lat={w[0]} w_cost={w[1]} w_energy={w[2]}")
    print(f"  Bornes: lat_ub={bounds.lat_ub} cost_max={bounds.cost_max} energy_ub={bounds.energy_ub}")
    
    # Afficher les solveurs activés
    solvers_status = []
    solvers_status.append(f"Greedy: {'ON' if not args.skip_greedy else 'OFF'}")
    solvers_status.append(f"CSP: {'ON' if not args.skip_csp else 'OFF'}")
    solvers_status.append(f"LLM: {'ON' if not args.skip_llm else 'OFF'}")
    print(f"  Solveurs: {' | '.join(solvers_status)}")
    if not args.skip_greedy:
        print(f"  Greedy mode: {args.greedy_mode}")
    if not args.skip_llm:
        print(f"  Seeds LLM: {'oui' if args.use_seeds else 'non (exploration pure)'}")

    results = []

    # --- 1. GLOUTON (FFD) ---
    if not args.skip_greedy:
        if args.greedy_mode == "energy":
            print("\n[2/4] Glouton-E (FFD energie)...")
            t0 = time.time()
            g_pl = greedy_energy_placement(infra, appli)
            g_name = "Glouton-E (FFD)"
        else:
            print("\n[2/4] Glouton Legacy (FFD cout+lat)...")
            t0 = time.time()
            g_pl = greedy_placement_legacy(infra, appli)
            g_name = "Glouton-L (FFD)"
        g_sol = evaluate_solution(g_pl, infra, appli, bounds, *w)
        t_ffd = time.time() - t0
        g_sol.solve_time = t_ffd
        print(f"  Obj={g_sol.objective_int} Lat_e2e={g_sol.total_latency}ms "
              f"P_total={g_sol.p_total_w:.2f}W (P_link={g_sol.p_link_w:.2f} P_node={g_sol.p_node_w:.2f}) "
              f"t={t_ffd:.4f}s")
        e_greedy_wh = P_CPU_SOLVER_W * t_ffd / 3600.0
        results.append(SolverResult(
            name=g_name, sol=g_sol,
            t_total_s=t_ffd,
            e_solver_wh=e_greedy_wh,
            e_solver_wh_min=e_greedy_wh * 0.8,
            e_solver_wh_max=e_greedy_wh * 1.2,
            e_method=f"P*T ({P_CPU_SOLVER_W:.0f}W) locale"))
    else:
        print("\n[2/4] Greedy: SKIP (--skip-greedy)")

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
                             max_iter=args.llm_iter, verbose=True,
                             use_seeds=args.use_seeds)
        results.append(llm_result)
    else:
        print("\n[4/4] LLM: SKIP (--skip-llm)")

    # --- RAPPORT ---
    print_comparison(results, bounds, infra, appli)


if __name__ == "__main__":
    main()
