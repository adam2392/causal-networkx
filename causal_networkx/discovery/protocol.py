from typing import Protocol
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Set, Union

import networkx as nx
from networkx.classes.reportviews import EdgeView, NodeView
from numpy.typing import NDArray



class GraphProtocol(Protocol):
    @property
    def nodes(self) -> NodeView:

    def adjacencies(self, u) -> List[Any]:

    def has_adjacency(self, u, v) -> bool:

    @abstractmethod
    def add_node(self, node_for_adding, **attr):
        pass

    @abstractmethod
    def remove_node(self, u):
        pass