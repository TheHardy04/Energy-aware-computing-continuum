from dataclasses import dataclass
from typing import Dict, Any, Optional, Protocol, Tuple, List


@dataclass
class PlacementResult:
    """
    Result of a placement algorithm, including:
        - mapping: Dict[int, int] mapping service IDs to host node IDs
        - paths: Dict[Tuple[int, int], List[int]] mapping service edges (u,v) to the list of infra node IDs representing the chosen path
        - meta: Dict[str, Any] for diagnostics, e.g., path info, resource usage, success/failure status, etc.
    """

    # Mapping from service id -> host node id
    mapping: Dict[int, int]

    # paths: mapping (u,v) -> list of infra node ids representing the chosen path
    paths: Dict[Tuple[int, int], List[int]]

    # diagnostics (e.g., path info, resource usage)
    meta: Dict[str, Any] 

