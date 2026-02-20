import json
from typing import Dict, Any, List, Optional

import networkx as nx


class ServiceGraph:
	"""Directed service graph built from AppProperties dict.

	Expected dict shape (from AppProperties.to_dict()):
	  {
		'application.nb': int,
		'application.components': int,
		'components': [ {'cpu': int, 'ram': int, 'lambda': int, 'mu': int}, ... ],
		'links': [ {'id': int, 'src': int, 'dst': int, 'bandwidth': int, 'latency': int}, ... ],
		'links.nb': int,
		'component.nbDZ': Optional[int],
		'component.DZ': List[int]
	  }
	"""

	def __init__(self):
		self.G = nx.DiGraph()
		self.metadata: Dict[str, Any] = {}

	@classmethod
	def from_app_dict(cls, app: Dict[str, Any]):
		obj = cls()
		obj.metadata['application.nb'] = app.get('application.nb')
		obj.metadata['application.components'] = app.get('application.components')
		obj.metadata['links.nb'] = app.get('links.nb')
		obj.metadata['component.nbDZ'] = app.get('component.nbDZ')
		obj.metadata['component.DZ'] = app.get('component.DZ')

		# Nodes: components
		components: List[Dict[str, Any]] = app.get('components', [])
		for comp_id, comp in enumerate(components):
			obj.G.add_node(
				comp_id,
				cpu=comp.get('cpu'),
				ram=comp.get('ram'),
				lambd=comp.get('lambda'),
				mu=comp.get('mu'),
			)

		# Edges: service links
		links: List[Dict[str, Any]] = app.get('links', [])
		for link in links:
			obj.G.add_edge(
				int(link['src']),
				int(link['dst']),
				id=int(link.get('id', -1)),
				bandwidth=int(link.get('bandwidth', 0)),
				latency=int(link.get('latency', 0)),
			)
		# Check that it is a DAG 
		if not nx.is_directed_acyclic_graph(obj.G):
			raise ValueError("Service graph must be a directed acyclic graph (DAG).")
		
		return obj

	# -------- info helpers ---------
	def summary(self) -> Dict[str, Any]:
		return {
			'nodes': self.G.number_of_nodes(),
			'edges': self.G.number_of_edges(),
			'is_directed': self.G.is_directed(),
			'application.components_meta': self.metadata.get('application.components'),
			'links.nb_meta': self.metadata.get('links.nb'),
		}

	def print_summary(self):
		print(json.dumps(self.summary(), indent=2))

	def print_nodes(self):
		for n, d in self.G.nodes(data=True):
			print(
				f"Component {n}: cpu={d.get('cpu')}, ram={d.get('ram')}, lambda={d.get('lambd')}, mu={d.get('mu')}"
			)

	def print_edges(self):
		for u, v, d in self.G.edges(data=True):
			print(
				f"{u} -> {v} (id={d.get('id')}): bandwidth={d.get('bandwidth')}, latency={d.get('latency')}"
			)

	def degree_stats(self) -> Dict[str, Any]:
		indeg = dict(self.G.in_degree())
		outdeg = dict(self.G.out_degree())
		return {
			'in_degree': indeg,
			'out_degree': outdeg,
			'max_in_degree': max(indeg.values()) if indeg else 0,
			'max_out_degree': max(outdeg.values()) if outdeg else 0,
		}

	def connectivity_info(self) -> Dict[str, Any]:
		info: Dict[str, Any] = {}
		if self.G.is_directed():
			info['strongly_connected'] = nx.is_strongly_connected(self.G) if self.G.number_of_nodes() > 0 else False
			info['weakly_connected'] = nx.is_weakly_connected(self.G) if self.G.number_of_nodes() > 0 else False
			info['num_weakly_components'] = (
				nx.number_weakly_connected_components(self.G) if self.G.number_of_nodes() > 0 else 0
			)
		else:
			info['connected'] = nx.is_connected(self.G) if self.G.number_of_nodes() > 0 else False
			info['num_components'] = nx.number_connected_components(self.G) if self.G.number_of_nodes() > 0 else 0
		return info

	# -------- visualization ---------
	def draw(
		self,
		with_labels: bool = True,
		layout: str = 'spring',
		show_edge_labels: bool = True,
		show_node_info_labels: bool = True,
		block: bool = True,
	):
		try:
			import matplotlib.pyplot as plt
		except Exception as e:
			raise RuntimeError("matplotlib is required for drawing. Install it or disable draw().") from e

		if not block:
			plt.figure()

		if layout == 'spring':
			pos = nx.spring_layout(self.G, seed=42)
		elif layout == 'kamada_kawai':
			pos = nx.kamada_kawai_layout(self.G)
		elif layout == 'circular':
			pos = nx.circular_layout(self.G)
		else:
			pos = nx.spring_layout(self.G, seed=42)

		# no self-links in service links by definition, but we filter just in case
		edgelist = [(u, v) for u, v in self.G.edges() if u != v]

		with_labels_draw = False if show_node_info_labels else with_labels
		nx.draw(
			self.G,
			pos,
			with_labels=with_labels_draw,
			node_color='#a4c2f4',
			node_size=800,
			arrows=True,
			edgelist=edgelist,
		)

		if show_node_info_labels:
			labels_nodes = {
				n: f"{n}\nCPU={d.get('cpu')}\nRAM={d.get('ram')}\nλ={d.get('lambd')}\nμ={d.get('mu')}"
				for n, d in self.G.nodes(data=True)
			}
			nx.draw_networkx_labels(self.G, pos, labels=labels_nodes, font_size=8)

		if show_edge_labels:
			labels = {
				(u, v): f"id={d.get('id')}, bw={d.get('bandwidth')}, lat={d.get('latency')}"
				for u, v, d in self.G.edges(data=True)
				if (u, v) in edgelist
			}
			nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels, font_size=8)

		plt.tight_layout()
		plt.show(block=block)

