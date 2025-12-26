import json
from typing import Dict, Any, List, Optional

import networkx as nx


class NetworkGraph:
	"""Graph wrapper built from InfraProperties dict.

	Expected infra dict shape (from InfraProperties.to_dict()):
	  {
		'hosts.nb': int,
		'hosts': [ {'cpu': int, 'ram': int}, ... ],
		'links': [ {'src': int, 'dst': int, 'bandwidth': int, 'latency': int}, ... ],
		'edges.nb': int,
		'network.diameter': int
	  }
	"""

	def __init__(self):
		self.G = nx.DiGraph()
		self.metadata: Dict[str, Any] = {}

	@classmethod
	def from_infra_dict(cls, infra: Dict[str, Any]):
		obj = cls()
		obj.metadata['hosts.nb'] = infra.get('hosts.nb')
		obj.metadata['edges.nb'] = infra.get('edges.nb')
		obj.metadata['network.diameter'] = infra.get('network.diameter')

		# nodes
		hosts: List[Dict[str, Any]] = infra.get('hosts', [])
		for node_id, host in enumerate(hosts):
			obj.G.add_node(node_id, cpu=host.get('cpu'), ram=host.get('ram'))

		# edges
		links: List[Dict[str, Any]] = infra.get('links', [])
		for link in links:
			obj.G.add_edge(
				int(link['src']),
				int(link['dst']),
				bandwidth=int(link.get('bandwidth', 0)),
				latency=int(link.get('latency', 0)),
			)
		return obj

	# -------- info helpers ---------
	def summary(self) -> Dict[str, Any]:
		return {
			'nodes': self.G.number_of_nodes(),
			'edges': self.G.number_of_edges(),
			'is_directed': self.G.is_directed(),
			'hosts.nb_meta': self.metadata.get('hosts.nb'),
			'edges.nb_meta': self.metadata.get('edges.nb'),
			'network.diameter_meta': self.metadata.get('network.diameter'),
		}

	def print_summary(self):
		s = self.summary()
		print(json.dumps(s, indent=2))

	def print_nodes(self):
		for n, data in self.G.nodes(data=True):
			print(f"Node {n}: cpu={data.get('cpu')}, ram={data.get('ram')}")

	def get_nodes_info(self) -> Dict[int, Dict[str, Any]]:
		"""Return a mapping node_id -> {'cpu': int|None, 'ram': int|None}."""
		return {n: {'cpu': d.get('cpu'), 'ram': d.get('ram')} for n, d in self.G.nodes(data=True)}

	def print_edges(self):
		for u, v, data in self.G.edges(data=True):
			print(
				f"{u} -> {v}: bandwidth={data.get('bandwidth')}, latency={data.get('latency')}"
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
			if nx.is_weakly_connected(self.G):
				info['num_weakly_components'] = nx.number_weakly_connected_components(self.G)
			else:
				info['num_weakly_components'] = nx.number_weakly_connected_components(self.G)
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
	):
		try:
			import matplotlib.pyplot as plt
		except Exception as e:
			raise RuntimeError("matplotlib is required for drawing. Install it or disable draw().") from e

		if layout == 'spring':
			pos = nx.spring_layout(self.G, seed=42)
		elif layout == 'kamada_kawai':
			pos = nx.kamada_kawai_layout(self.G)
		elif layout == 'circular':
			pos = nx.circular_layout(self.G)
		else:
			pos = nx.spring_layout(self.G, seed=42)

		# choose which edges to draw 
		edgelist = [(u, v) for u, v in self.G.edges() if u != v]

		# If we want to show node info (CPU/RAM), we'll draw custom labels
		with_labels_draw = False if show_node_info_labels else with_labels

		# Compute node colors: map to 3 colors depending on CPU value
		palette = ['#8fce00', '#ffd966', '#e06666']  # green, yellow, red
		t1, t2 =  (4, 8)
		node_colors = []
		for _, data in self.G.nodes(data=True):
			val = data.get('cpu', 0)
			if val <= t1:
				node_colors.append(palette[0])
			elif val <= t2:
				node_colors.append(palette[1])
			else:
				node_colors.append(palette[2])


		# draw the nodes and edges
		nx.draw(self.G, pos, with_labels=with_labels_draw, node_color=node_colors, node_size=1000, arrows=True, edgelist=edgelist)

		if show_node_info_labels:
			labels_nodes = {
				n: f"{n}\nCPU={d.get('cpu')}\nRAM={d.get('ram')}" for n, d in self.G.nodes(data=True)
			}
			nx.draw_networkx_labels(self.G, pos, labels=labels_nodes, font_size=8)
		if show_edge_labels:
			labels = {
				(u, v): f"bw={d.get('bandwidth')}, lat={d.get('latency')}"
				for u, v, d in self.G.edges(data=True)
				if (u != v) and ((u, v) in edgelist)
			}
			nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels, font_size=8)

		plt.tight_layout()
		plt.show()


