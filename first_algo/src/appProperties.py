import javaproperties

import re
import json
from typing import List, Dict, Any, Optional


def parse_braced_tuples(s: str) -> List[List[str]]:
	items = re.findall(r"\{([^}]*)\}", s)
	parsed: List[List[str]] = []
	for it in items:
		parts = [p.strip() for p in it.split(',') if p.strip() != '']
		parsed.append(parts)
	return parsed


class AppProperties:
	"""Parser for the application .properties content.

	Expects keys similar to `properties/Appli_4comps.properties`:
	  - application.nb
	  - application.components
	  - components.requirements: list of {CPU,RAM,lambda,mu}
	  - links.description: list of {id,src,dst,bandwidth,latency}
	  - links.nb
	  - component.nbDZ (optional)
	  - component.DZ (optional list)
	"""

	def __init__(self, props: Dict[str, str]):
		self.props = props
		self.application_nb: int = 0
		self.components_count: int = 0

		# Components: list of dicts {cpu, ram, lambda, mu}
		self.components: List[Dict[str, Any]] = []

		# Links: list of dicts {id, src, dst, bandwidth, latency}
		self.links: List[Dict[str, Any]] = []
		self.links_nb: int = 0

		# Locality constraints (optional)
		self.component_nbDZ: Optional[int] = None
		self.component_DZ: List[int] = []

		self._parse_all()

	@classmethod
	def from_file(cls, file_path: str = 'properties/Appli_4comps.properties'):
		with open(file_path, 'rb') as f:
			props = javaproperties.load(f)
		return cls(props)

	def _parse_all(self):
		self.application_nb = int(self.props.get('application.nb', 0))
		self.components_count = int(self.props.get('application.components', 0))
		self._parse_components()
		self._parse_links()
		self._parse_constraints()

	def _parse_components(self):
		raw = self.props.get('components.requirements', '')
		entries = parse_braced_tuples(raw)
		comps: List[Dict[str, Any]] = []
		for e in entries:
			# Expect [CPU, RAM, lambda, mu]
			comp = {
				'cpu': int(e[0]) if len(e) > 0 else None,
				'ram': int(e[1]) if len(e) > 1 else None,
				'lambda': int(e[2]) if len(e) > 2 else None,
				'mu': int(e[3]) if len(e) > 3 else None,
			}
			comps.append(comp)
		self.components = comps

	def _parse_links(self):
		raw = self.props.get('links.description', '')
		entries = parse_braced_tuples(raw)
		links: List[Dict[str, Any]] = []
		for e in entries:
			# Expect [id, src, dst, bandwidth, latency]
			if len(e) >= 5:
				link = {
					'id': int(e[0]),
					'src': int(e[1]),
					'dst': int(e[2]),
					'bandwidth': int(e[3]),
					'latency': int(e[4]),
				}
				links.append(link)
		self.links = links
		self.links_nb = int(self.props.get('links.nb', len(links)))

	def _parse_constraints(self):
		if 'component.nbDZ' in self.props:
			self.component_nbDZ = int(self.props.get('component.nbDZ', 0))
		raw = self.props.get('component.DZ', '')
		if raw:
			# component.DZ may be a list of integers in one brace, e.g., {0, 5}
			entries = parse_braced_tuples(raw)
			if entries:
				try:
					self.component_DZ = [int(x) for x in entries[0]]
				except Exception:
					self.component_DZ = []

	def to_dict(self) -> Dict[str, Any]:
		return {
			'application.nb': self.application_nb,
			'application.components': self.components_count,
			'components': self.components,
			'links': self.links,
			'links.nb': self.links_nb,
			'component.nbDZ': self.component_nbDZ,
			'component.DZ': self.component_DZ,
		}

	def to_json(self, **kwargs) -> str:
		return json.dumps(self.to_dict(), **kwargs)

