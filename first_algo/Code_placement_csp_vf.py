"""
Code_Placement_CSP - Placement via CP-SAT (OR-Tools)
Optimisation conjointe: Cout infra + Latence + Energie
"""

import re, time
from typing import List, Tuple, Dict, Set
from collections import Counter
from ortools.sat.python import cp_model

P_STATIC = 200
P_CPU_UNIT = 5


def parse_properties(path):
    props = {}
    with open(path, encoding="utf-8") as f:
        cont = ""
        for raw in f:
            line = raw.rstrip("\n")
            if line.endswith("\\"):
                cont += line[:-1]; continue
            line = (cont + line).strip(); cont = ""
            if not line or line[0] in "#!": continue
            if "=" in line: k, v = line.split("=", 1)
            elif ":" in line: k, v = line.split(":", 1)
            else: continue
            props[k.strip()] = v.strip()
    return props


def extract_ints(s):
    return list(map(int, re.findall(r"-?\d+", s)))


class BestSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, v, cpu_caps, ram_caps):
        super().__init__()
        self._v = v; self.cnt = 0; self.best = float("inf"); self.t0 = time.time()
        self.cpu_caps = cpu_caps; self.ram_caps = ram_caps

    def OnSolutionCallback(self):
        obj = self.ObjectiveValue()
        if obj < self.best:
            self.best = obj; self.cnt += 1
            print(f"\n--- Solution #{self.cnt} ({time.time()-self.t0:.2f}s) obj={obj:.0f} ---")
            for k in ["cout", "latence_max", "latence_tot", "energie"]:
                if k in self._v and self._v[k] is not None:
                    print(f"  {k}: {self.Value(self._v[k])}")
            for i, (vc, vr) in enumerate(zip(self._v.get("hcpu",[]), self._v.get("hram",[]))):
                c = self.Value(vc)
                if c > 0:
                    print(f"  H{i}: CPU {c}/{self.cpu_caps[i]} RAM {self.Value(vr)}/{self.ram_caps[i]}")


class TPDS2_Placement:
    def __init__(self):
        self.LOOP = -99
        self.nbH = self.FHOST = self.diameter = self.nbA = self.FARC = 0
        self.cpuCap = []; self.ramCap = []; self.hostCost = []
        self.connections = []; self.bwsCap = []; self.latencies = []
        self.arcs = []; self.linkInfra = []
        self.nbS = self.nbC = self.nbL = self.nbDZ = 0
        self.cpuComp = []; self.ramComp = []
        self.linkPerServ = []; self.latPair = []; self.bdwPair = []
        self.DZ = []; self.dataFlowRates = []; self.muRates = []
        self.model = None
        self.h = []; self.n = []; self.a = []; self.use = []
        self.hcpu = []; self.hram = []; self.total_flow_on_arc = []
        self.loop = set(); self.unlimited = set()
        self.uv2arc = {}; self.fake_arc_list = []; self.closure_pairs = []
        self.lats_tab = []; self.ConsumedLat = []
        self.total_deployment_cost = None; self.worst_case_latency = None
        self.total_latency = None; self.total_energy_global = None
        self.total_link_energy = None; self.total_node_energy = None
        self.final_objective = None

    # === LOAD ===
    def load_data(self, infra_file, appli_file):
        self._load_infra(infra_file)
        self._load_appli(appli_file)
        self._build_arcs()

    def _load_infra(self, f):
        p = parse_properties(f)
        self.nbH = int(p["hosts.nb"])
        self.diameter = int(p.get("network.diameter", self.nbH - 1))
        cfg = extract_ints(p["hosts.configuration"])
        self.cpuCap = [cfg[2*i] for i in range(self.nbH)]
        self.ramCap = [cfg[2*i+1] for i in range(self.nbH)]
        self.hostCost = extract_ints(p["hosts.cost"])[:self.nbH] if "hosts.cost" in p else list(self.cpuCap)
        while len(self.hostCost) < self.nbH:
            self.hostCost.append(self.cpuCap[len(self.hostCost)])
        if "hosts.cost" not in p:
            print("INFO: hostCost = cpuCap (fallback)")

        base = int(p["edges.nb"])
        self.connections = [[-1]*self.nbH for _ in range(self.nbH)]
        self.bwsCap = [0]*base; self.latencies = [0]*base
        topo = extract_ints(p["network.topology"])
        arc = 0
        for i in range(0, len(topo), 4):
            u, v, bw, lat = topo[i:i+4]
            self.connections[u][v] = arc
            self.linkInfra.append([arc, u, v])
            self.bwsCap[arc] = bw if u != v else self.LOOP
            self.latencies[arc] = lat if u != v else 0
            arc += 1
        self.nbA = base

    def _load_appli(self, f):
        p = parse_properties(f)
        comps = extract_ints(p["application.components"])
        self.nbS = int(p["application.nb"]); self.nbC = sum(comps)
        reqs = extract_ints(p["components.requirements"])
        self.cpuComp = [reqs[4*i] for i in range(self.nbC)]
        self.ramComp = [reqs[4*i+1] for i in range(self.nbC)]
        self.dataFlowRates = [reqs[4*i+2] for i in range(self.nbC)]
        self.muRates = [reqs[4*i+3] for i in range(self.nbC)]
        links = extract_ints(p["links.description"])
        self.nbL = int(p["links.nb"])
        for i in range(self.nbL):
            _, s, d, bw, lat = links[5*i:5*i+5]
            self.linkPerServ.append((s, d)); self.bdwPair.append(bw); self.latPair.append(lat)
        self.nbDZ = int(p["component.nbDZ"]); self.DZ = []
        if self.nbDZ > 0:
            dz = extract_ints(p.get("component.DZ", ""))
            for i in range(self.nbDZ):
                self.DZ.append((dz[2*i], dz[2*i+1]))

    # === ARCS ===
    def _build_arcs(self):
        self.FHOST = self.nbH; self.loop = set(); self.arcs = []
        for i in range(self.nbH):
            for j in range(self.nbH):
                if self.connections[i][j] > -1:
                    aid = self.connections[i][j]
                    self.arcs.append((i, aid, j))
                    if i == j: self.loop.add(aid)
        for i in range(self.nbH):
            aid = self.nbA; self.nbA += 1
            while len(self.bwsCap) < self.nbA: self.bwsCap.append(0)
            while len(self.latencies) < self.nbA: self.latencies.append(0)
            self.bwsCap[aid] = self.LOOP; self.latencies[aid] = 0
            self.arcs.append((i, aid, self.FHOST))
        aid = self.nbA; self.nbA += 1
        while len(self.bwsCap) < self.nbA: self.bwsCap.append(0)
        while len(self.latencies) < self.nbA: self.latencies.append(0)
        self.bwsCap[aid] = self.LOOP; self.latencies[aid] = 0
        self.arcs.append((self.FHOST, aid, self.FHOST)); self.loop.add(aid)
        self.FARC = self.nbA
        self.uv2arc = {(u, v): aid for (u, aid, v) in self.arcs}
        self.fake_arc_list = [self.uv2arc[(i, self.FHOST)] for i in range(self.nbH)]
        self.fake_arc_list.append(self.uv2arc[(self.FHOST, self.FHOST)])
        self.closure_pairs = []
        for h in range(self.nbH):
            if (h, self.FHOST) in self.uv2arc:
                self.closure_pairs.append([h, self.uv2arc[(h, self.FHOST)]])
        self.closure_pairs.append([self.FHOST, self.uv2arc[(self.FHOST, self.FHOST)]])
        self.lats_tab = self.latencies + [0]  # index FARC -> 0
        print(f"INFO: nbA={self.nbA} FARC={self.FARC} FHOST={self.FHOST} arcs={len(self.arcs)}")

    # === VARIABLES ===
    def declare_variables(self):
        m = self.model = cp_model.CpModel()
        D = self.diameter
        self.h = [m.NewIntVar(0, self.nbH-1, f"h_{c}") for c in range(self.nbC)]
        self.n, self.a, self.use = [], [], []
        for l in range(self.nbL):
            self.n.append([m.NewIntVar(0, self.FHOST, f"n_{l}_{k}") for k in range(D+1)])
            self.a.append([m.NewIntVar(0, self.FARC, f"a_{l}_{k}") for k in range(D)])
            self.use.append([m.NewBoolVar(f"u_{l}_{k}") for k in range(D)])
        self.hcpu = [m.NewIntVar(0, self.cpuCap[h], f"hC_{h}") for h in range(self.nbH)]
        self.hram = [m.NewIntVar(0, self.ramCap[h], f"hR_{h}") for h in range(self.nbH)]
        max_flow = max(1, sum(self.bdwPair))
        self.total_flow_on_arc = [m.NewIntVar(0, max_flow, f"f_{e}") for e in range(self.nbA+1)]

    # === CONSTRAINTS ===
    def declare_constraints(self):
        self.ConsumedLat = [None]*self.nbL
        for c, h in self.DZ:
            if 0 <= c < self.nbC and 0 <= h < self.nbH:
                self.model.Add(self.h[c] == h)
        self._resource_constraints()
        for l in range(self.nbL):
            self._path_constraint(l)
        self._flow_constraints()

    def _resource_constraints(self):
        m = self.model
        for k in range(self.nbH):
            cpu_t, ram_t = [], []
            for c in range(self.nbC):
                on = m.NewBoolVar(f"o_{c}_{k}")
                m.Add(self.h[c] == k).OnlyEnforceIf(on)
                m.Add(self.h[c] != k).OnlyEnforceIf(on.Not())
                cd = m.NewIntVar(0, self.cpuComp[c], f"cd_{c}_{k}")
                rd = m.NewIntVar(0, self.ramComp[c], f"rd_{c}_{k}")
                m.Add(cd == self.cpuComp[c]).OnlyEnforceIf(on)
                m.Add(cd == 0).OnlyEnforceIf(on.Not())
                m.Add(rd == self.ramComp[c]).OnlyEnforceIf(on)
                m.Add(rd == 0).OnlyEnforceIf(on.Not())
                cpu_t.append(cd); ram_t.append(rd)
            m.Add(sum(cpu_t) <= self.cpuCap[k])
            m.Add(sum(ram_t) <= self.ramCap[k])
            m.Add(self.hcpu[k] == sum(cpu_t))
            m.Add(self.hram[k] == sum(ram_t))

    def _path_constraint(self, l):
        m, D = self.model, self.diameter
        n, a, use = self.n[l], self.a[l], self.use[l]
        sC, tC = self.linkPerServ[l]
        s_h, t_h = self.h[sC], self.h[tC]

        same = m.NewBoolVar(f"sm_{l}")
        m.Add(s_h == t_h).OnlyEnforceIf(same)
        m.Add(s_h != t_h).OnlyEnforceIf(same.Not())

        for k in range(D-1): m.Add(use[k] >= use[k+1])
        m.Add(n[0] == s_h)

        # same host: all use=0, immediate closure
        for k in range(D): m.Add(use[k] == 0).OnlyEnforceIf(same)

        # different: at least 1 hop
        m.Add(use[0] >= 1).OnlyEnforceIf(same.Not())

        for k in range(D):
            # real hop
            ct = m.AddAllowedAssignments([n[k], a[k], n[k+1]], self.arcs)
            ct.OnlyEnforceIf(use[k])
            m.Add(n[k] != self.FHOST).OnlyEnforceIf(use[k])
            m.Add(n[k+1] != self.FHOST).OnlyEnforceIf(use[k])
            # closure
            cc = m.AddAllowedAssignments([n[k], a[k]], self.closure_pairs)
            cc.OnlyEnforceIf(use[k].Not())
            m.Add(n[k+1] == self.FHOST).OnlyEnforceIf(use[k].Not())

        # last hop -> target
        for j in range(D):
            il = m.NewBoolVar(f"il_{l}_{j}")
            if j == D-1:
                m.Add(il == use[j])
            else:
                m.AddBoolAnd([use[j], use[j+1].Not()]).OnlyEnforceIf(il)
                m.AddBoolOr([use[j].Not(), use[j+1]]).OnlyEnforceIf(il.Not())
            m.Add(n[j+1] == t_h).OnlyEnforceIf([il, same.Not()])

        # anti-cycle
        for i in range(D+1):
            for j in range(i+1, D+1):
                conds = [same.Not()]
                if i > 0: conds.append(use[i-1])
                if j > 0: conds.append(use[j-1])
                m.Add(n[i] != n[j]).OnlyEnforceIf(conds)

        # latency (unconditional element — closure arcs have lat=0)
        max_al = max(self.lats_tab) if self.lats_tab else 0
        lats = []
        for k in range(D):
            lk = m.NewIntVar(0, max_al, f"lt_{l}_{k}")
            m.AddElement(a[k], self.lats_tab, lk)
            lats.append(lk)
        tl = m.NewIntVar(0, D * max_al, f"tl_{l}")
        m.Add(tl == sum(lats))
        self.ConsumedLat[l] = tl

    def _flow_constraints(self):
        m, D = self.model, self.diameter
        fa = [[] for _ in range(self.nbA + 1)]
        for l in range(self.nbL):
            bw = self.bdwPair[l]
            if bw == 0: continue
            for k in range(D):
                sf = m.NewIntVar(0, bw, f"sf_{l}_{k}")
                m.Add(sf == bw).OnlyEnforceIf(self.use[l][k])
                m.Add(sf == 0).OnlyEnforceIf(self.use[l][k].Not())
                for e in range(self.nbA + 1):
                    ct = m.NewIntVar(0, bw, f"c_{l}_{k}_{e}")
                    ie = m.NewBoolVar(f"i_{l}_{k}_{e}")
                    m.Add(self.a[l][k] == e).OnlyEnforceIf(ie)
                    m.Add(self.a[l][k] != e).OnlyEnforceIf(ie.Not())
                    m.Add(ct == sf).OnlyEnforceIf(ie)
                    m.Add(ct == 0).OnlyEnforceIf(ie.Not())
                    fa[e].append(ct)
        for e in range(self.nbA + 1):
            if fa[e]:
                m.Add(self.total_flow_on_arc[e] == sum(fa[e]))
            else:
                m.Add(self.total_flow_on_arc[e] == 0)
            if (e < len(self.bwsCap) and e not in self.loop
                    and self.bwsCap[e] != self.LOOP and self.bwsCap[e] > 0):
                m.Add(self.total_flow_on_arc[e] <= self.bwsCap[e])

    # === OBJECTIVE ===
    def build_objective(self, w_lat=0, w_cost=0, w_energy=1):
        m = self.model

        # --- Latence ---
        all_lats = [x for x in self.ConsumedLat if x is not None]
        max_al = max(self.latencies) if self.latencies else 1
        lat_ub = max(1, self.diameter * max_al)
        if all_lats:
            self.worst_case_latency = m.NewIntVar(0, lat_ub, "wc_lat")
            m.AddMaxEquality(self.worst_case_latency, all_lats)
        else:
            self.worst_case_latency = m.NewConstant(0)
        self.total_latency = m.NewIntVar(0, self.nbL * lat_ub, "tot_lat")
        m.Add(self.total_latency == (sum(all_lats) if all_lats else 0))

        # --- Cout infra ---
        cost_max = max(1, sum(self.hostCost))
        hct = []
        for h in range(self.nbH):
            ia = m.NewBoolVar(f"ca_{h}")
            m.Add(self.hcpu[h] > 0).OnlyEnforceIf(ia)
            m.Add(self.hcpu[h] == 0).OnlyEnforceIf(ia.Not())
            hc = m.NewIntVar(0, self.hostCost[h], f"hc_{h}")
            m.Add(hc == self.hostCost[h]).OnlyEnforceIf(ia)
            m.Add(hc == 0).OnlyEnforceIf(ia.Not())
            hct.append(hc)
        self.total_deployment_cost = m.NewIntVar(0, cost_max, "tc")
        m.Add(self.total_deployment_cost == sum(hct))

        # --- Energie ---
        # E_link = sum(flow_arc * lat_arc^2)
        max_flow = max(1, sum(self.bdwPair))
        max_arc_lat = max((lat for lat in self.latencies if lat > 0), default=1)
        e_link_ub = max(1, max_flow * max_arc_lat * max_arc_lat)

        lt = []
        for e in range(self.nbA):
            lat = self.latencies[e] if e < len(self.latencies) else 0
            if lat > 0:
                lt.append(self.total_flow_on_arc[e] * (lat * lat))
        self.total_link_energy = m.NewIntVar(0, e_link_ub, "e_link")
        m.Add(self.total_link_energy == (sum(lt) if lt else 0))

        # E_node = sum(P_STATIC + cpu_used * P_CPU_UNIT) for active hosts
        e_node_ub = self.nbH * (P_STATIC + max(self.cpuCap) * P_CPU_UNIT)
        nt = []
        for h in range(self.nbH):
            ia = m.NewBoolVar(f"ea_{h}")
            m.Add(self.hcpu[h] > 0).OnlyEnforceIf(ia)
            m.Add(self.hcpu[h] == 0).OnlyEnforceIf(ia.Not())
            st = m.NewIntVar(0, P_STATIC, f"es_{h}")
            m.Add(st == P_STATIC).OnlyEnforceIf(ia)
            m.Add(st == 0).OnlyEnforceIf(ia.Not())
            dy = m.NewIntVar(0, self.cpuCap[h] * P_CPU_UNIT, f"ed_{h}")
            m.Add(dy == self.hcpu[h] * P_CPU_UNIT)
            nt.extend([st, dy])
        self.total_node_energy = m.NewIntVar(0, e_node_ub, "e_node")
        m.Add(self.total_node_energy == sum(nt))

        energy_ub = e_link_ub + e_node_ub
        self.total_energy_global = m.NewIntVar(0, energy_ub, "e_tot")
        m.Add(self.total_energy_global == self.total_link_energy + self.total_node_energy)

        # --- Objectif normalisé (scaled integer) ---
        # On utilise NORM = 10000 pour eviter tout overflow
        NORM = 10000

        def normalize(var, ub, name):
            """Returns IntVar in [0, NORM] = var * NORM / ub."""
            if ub <= 0: return m.NewConstant(0)
            num = m.NewIntVar(0, ub * NORM, f"{name}_n")
            m.Add(num == var * NORM)
            norm = m.NewIntVar(0, NORM, f"{name}_r")
            m.AddDivisionEquality(norm, num, ub)
            return norm

        lat_n = normalize(self.worst_case_latency, lat_ub, "lat")
        cost_n = normalize(self.total_deployment_cost, cost_max, "cost")
        ener_n = normalize(self.total_energy_global, energy_ub, "ener")

        W = 1000
        wl = int(round(w_lat * W))
        wc = int(round(w_cost * W))
        we = int(round(w_energy * W))

        obj = m.NewIntVar(0, NORM * W, "obj")
        m.Add(obj == lat_n * wl + cost_n * wc + ener_n * we)
        m.Minimize(obj)
        self.final_objective = obj

        print(f"INFO: 3 objectifs w_lat={w_lat} w_cost={w_cost} w_energy={w_energy}")
        print(f"INFO: Bornes lat_ub={lat_ub} cost_max={cost_max} energy_ub={energy_ub}")

    # === PRINTS ===
    def print_data_summary(self):
        print(f"\n{'='*60}\n DONNEES\n{'='*60}")
        print(f"  Hotes: {self.nbH} | Diametre: {self.diameter}")
        print(f"  CPU:  {self.cpuCap}\n  RAM:  {self.ramCap}\n  Cout: {self.hostCost}")
        print(f"  Composants: {self.nbC} | Liens: {self.nbL}")
        print(f"  CPU comp: {self.cpuComp}\n  RAM comp: {self.ramComp}")
        print(f"  bdwPair:  {self.bdwPair}\n  latPair:  {self.latPair}")
        if self.DZ: print(f"  Localite: {self.DZ}")
        print("=" * 60)

    def print_model_stats(self):
        pr = self.model.Proto()
        nv = len(pr.variables)
        nb = sum(1 for v in pr.variables if list(v.domain) == [0,1])
        
        # Simplified: just count constraint types safely
        print(f"\nModele: {nv} vars ({nb} bool), {len(pr.constraints)} contraintes")
        print(f"  (constraint type breakdown skipped due to API compatibility)")


# === MAIN ===
def print_final_report(sol, solver, status, t_build, t_solve, t_total):
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        stat_name = solver.StatusName(status)
        tc = solver.Value(sol.total_deployment_cost)
        wl = solver.Value(sol.worst_case_latency)
        tl = solver.Value(sol.total_latency)
        en = solver.Value(sol.total_energy_global)
        el = solver.Value(sol.total_link_energy)
        eno = solver.Value(sol.total_node_energy)

        print(f"\n{'='*70}")
        print(f" RAPPORT FINAL - CP-SAT ({stat_name})")
        print(f"{'='*70}")

        # Tableau des metriques
        print(f"\n {'METRIQUES OPTIMISEES':^50}")
        print(f" {'-'*50}")
        print(f"  {'Metrique':<30} {'Valeur':>15}")
        print(f"  {'-'*30} {'-'*15}")
        print(f"  {'Latence bout en bout (max)':<30} {wl:>12} ms")
        print(f"  {'Latence totale (somme)':<30} {tl:>12} ms")
        print(f"  {'Cout de deploiement':<30} {tc:>15}")
        print(f"  {'Energie liens':<30} {el:>15}")
        print(f"  {'Energie noeuds':<30} {eno:>15}")
        print(f"  {'Energie TOTALE':<30} {en:>15}")
        print(f"  {'Fonction objectif':<30} {solver.ObjectiveValue():>15.2f}")
        print(f"  {'-'*30} {'-'*15}")

        # Temps
        print(f"\n {'TEMPS D EXECUTION':^50}")
        print(f" {'-'*50}")
        print(f"  {'Construction du modele':<30} {t_build:>12.3f} s")
        print(f"  {'Resolution CP-SAT':<30} {t_solve:>12.3f} s")
        print(f"  {'Temps TOTAL':<30} {t_total:>12.3f} s")
        print(f"  {'Nb solutions explorees':<30} {solver.NumBooleans():>15}")
        print(f"  {'-'*50}")

        # Hotes actifs
        active = []
        for h in range(sol.nbH):
            c = solver.Value(sol.hcpu[h])
            if c > 0: active.append(h)
        print(f"\n  Hotes actifs: {len(active)} / {sol.nbH}  ->  {active}")

        print(f"\n PLACEMENT:")
        for c in range(sol.nbC):
            print(f"  Comp {c} -> Hote {solver.Value(sol.h[c])}")

        print(f"\n CHEMINS:")
        for l in range(sol.nbL):
            sC, tC = sol.linkPerServ[l]
            sh = solver.Value(sol.h[sC]); th = solver.Value(sol.h[tC])
            path = []
            for k in range(sol.diameter + 1):
                nd = solver.Value(sol.n[l][k]); path.append(nd)
                if nd == sol.FHOST: break
            real = [nd for nd in path if nd != sol.FHOST]
            arcs = []
            for k in range(sol.diameter):
                if solver.Value(sol.use[l][k]): arcs.append(solver.Value(sol.a[l][k]))
                else: break
            lat = solver.Value(sol.ConsumedLat[l])
            print(f"  Lien {l} (C{sC}->C{tC}): H{sh}->H{th} | chemin={real} | arcs={arcs} | lat={lat} ms")

        print(f"\n RESSOURCES PAR HOTE:")
        print(f"  {'Hote':>5} | {'CPU':>10} | {'RAM':>14} | {'Cout':>6} | {'Taux CPU':>8}")
        print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*14}-+-{'-'*6}-+-{'-'*8}")
        for h in range(sol.nbH):
            c = solver.Value(sol.hcpu[h])
            if c > 0:
                r = solver.Value(sol.hram[h])
                pct = 100*c/sol.cpuCap[h]
                print(f"  H{h:>3} | {c:>4}/{sol.cpuCap[h]:<4} | {r:>6}/{sol.ramCap[h]:<6} | "
                      f"{sol.hostCost[h]:>6} | {pct:>6.0f} %")
        print(f"{'='*70}")
    else:
        print(f"\nPas de solution: {solver.StatusName(status)}")
        if status == cp_model.MODEL_INVALID:
            err = sol.model.Validate()
            print(f"  Erreur de validation: {err}")


def main():
    t_global = time.time()
    sol = TPDS2_Placement()
    print("Chargement...")
    try:
        # sol.load_data("properties/Infra_8nodes.properties", "properties/Appli_4comps_2.properties")
        sol.load_data("properties/Infra_16nodes_fog3tier.properties", "properties/Appli_8comps_smartbuilding.properties")
        
        # sol.load_data("resources/Infra_24nodes_dcns.properties", "resources/Appli_10comps_dcns.properties")
        #sol.load_data("resources/Infra_28nodes_smartcity.properties", "resources/Appli_11comps_smartcity.properties")
        
        #sol.load_data("resources/Infra_32nodes_hospital.properties", "resources/Appli_12comps_ehealth.properties")
        
    except Exception as e:
        print(f"Erreur: {e}"); return

    sol.print_data_summary()

    t_build_start = time.time()
    print("\nConstruction du modele...")
    sol.declare_variables()
    sol.declare_constraints()
    print("Objectif...")
    sol.build_objective(w_lat=0, w_cost=0, w_energy=1)
    sol.print_model_stats()
    t_build = time.time() - t_build_start

    err = sol.model.Validate()
    if err:
        print(f"\nERREUR MODELE: {err}"); return

    v = {"cout": sol.total_deployment_cost, "latence_max": sol.worst_case_latency,
         "latence_tot": sol.total_latency, "energie": sol.total_energy_global,
         "hcpu": sol.hcpu, "hram": sol.hram}
    cb = BestSolutionPrinter(v, sol.cpuCap, sol.ramCap)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8

    print("\nResolution...")
    t_solve_start = time.time()
    status = solver.Solve(sol.model, cb)
    t_solve = time.time() - t_solve_start
    t_total = time.time() - t_global

    print_final_report(sol, solver, status, t_build, t_solve, t_total)


if __name__ == "__main__":
    main()
