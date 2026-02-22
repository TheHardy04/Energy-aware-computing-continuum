import javaproperties

import re
import json

# helper to parse a list of braced tuples like: {16,32000}, {8,16000}
def parse_braced_tuples(s):
    # find all occurrences of { ... }
    items = re.findall(r"\{([^}]*)\}", s)
    parsed = []
    for it in items:
        parts = [p.strip() for p in it.split(',') if p.strip()!='']
        parsed.append(parts)
    return parsed


class InfraProperties:
    """Parser for the infra .properties content.

    Usage:
      infra = InfraProperties.from_file('properties/Infra_8nodes.properties')
      print(infra.to_json())
    """

    def __init__(self, props: dict):
        self.props = props
        self.hosts = []
        self.links = []
        self.edges_nb = 0
        self.network_diameter = None
        self.hosts_nb = 0
        self._parse_all()

    @classmethod
    def from_file(cls, file_path: str = 'properties/Infra_8nodes.properties'):
        with open(file_path, 'rb') as f:
            props = javaproperties.load(f)
        return cls(props)

    def _parse_all(self):
        self.hosts_nb = int(self.props.get('hosts.nb', 0))
        self._parse_hosts()
        self._parse_topology()
        self.edges_nb = int(self.props.get('edges.nb', 0))
        self.network_diameter = int(self.props.get('network.diameter', 0)) if 'network.diameter' in self.props else None

    def _parse_hosts(self):
        hosts_config_raw = self.props.get('hosts.configuration', '')
        hosts_parsed = parse_braced_tuples(hosts_config_raw)
        hosts = []
        for entry in hosts_parsed:
            cpu = int(entry[0]) if len(entry) > 0 else None
            ram = int(entry[1]) if len(entry) > 1 else None
            hosts.append({'cpu': cpu, 'ram': ram})
        self.hosts = hosts

    def _parse_topology(self):
        topology_raw = self.props.get('network.topology', '')
        topology_parsed = parse_braced_tuples(topology_raw)
        links = []
        for entry in topology_parsed:
            if len(entry) >= 4:
                src = int(entry[0])
                dst = int(entry[1])
                bw = int(entry[2])
                lat = int(entry[3])
                links.append({'src': src, 'dst': dst, 'bandwidth': bw, 'latency': lat})
        self.links = links

    def to_dict(self):
        return {
            'hosts.nb': self.hosts_nb,
            'hosts': self.hosts,
            'links': self.links,
            'edges.nb': self.edges_nb,
            'network.diameter': self.network_diameter,
        }

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)







