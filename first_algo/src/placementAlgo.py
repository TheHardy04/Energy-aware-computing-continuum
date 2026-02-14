from abc import ABC, abstractmethod


from src.base import PlacementResult
from src.networkGraph import NetworkGraph
from src.serviceGraph import ServiceGraph

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
    