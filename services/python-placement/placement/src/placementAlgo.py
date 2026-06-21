from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph

@dataclass
class PlacementResult:
    """
    Summary of a placement result.
    Attributes:
        mapping: Dict[int, int] - Mapping from service component id to infrastructure host id
        paths: Dict[Tuple[int, int], List[int]] - Mapping from service edge (src_comp_id, dst_comp_id) to list of infra node ids representing the chosen path
        meta: Dict[str, Any] - Optional dictionary for any additional metadata or diagnostics (e.g., resource usage, path details)
    """

    # Mapping from service id -> host node id
    mapping: Dict[int, int]

    # paths: mapping (u,v) -> list of infra node ids representing the chosen path
    paths: Dict[Tuple[int, int], List[int]]

    # diagnostics (e.g., path info, resource usage)
    meta: Dict[str, Any]

class PlacementAlgo(ABC):
    """Abstract base class for placement algorithms."""
    
    @abstractmethod
    def place(self, service_graph: ServiceGraph, network_graph: NetworkGraph, **kwargs) -> PlacementResult:
        """
        Place services from the service graph onto nodes in the network graph.
        
        Args:
            service_graph: The service dependency graph to be placed
            network_graph: The target network infrastructure
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            A placement solution in the type of PlacementResult, containing the mapping and any relevant metadata.
        """
        raise NotImplementedError("Subclasses must implement place()")
    