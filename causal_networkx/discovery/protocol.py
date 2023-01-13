from abc import abstractmethod
from typing import Any, List, Protocol

from networkx.classes.reportviews import EdgeView, NodeView


class GraphProtocol(Protocol):
    @property
    def nodes(self) -> NodeView:
        pass

    @property
    def edges(self) -> EdgeView:
        pass

    def adjacencies(self, u) -> List[Any]:
        pass

    def has_adjacency(self, u, v) -> bool:
        pass

    @abstractmethod
    def add_node(self, node_for_adding, **attr):
        pass

    @abstractmethod
    def remove_node(self, u):
        pass
