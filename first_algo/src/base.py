from dataclasses import dataclass
from typing import Dict, Any, Optional, Protocol, Tuple, List


@dataclass
class PlacementResult:
    # Mapping from service/component id -> host node id
    mapping: Dict[int, int]

    # paths: mapping (u,v) -> list of infra node ids representing the chosen path
    paths: Dict[Tuple[int, int], List[int]]

    # diagnostics (e.g., path info, resource usage)
    meta: Dict[str, Any] 

