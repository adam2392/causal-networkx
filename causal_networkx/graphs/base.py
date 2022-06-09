from abc import ABCMeta, abstractmethod
from typing import Any, List, Set, Union

import networkx as nx
from networkx.classes.reportviews import EdgeView, NodeView
from numpy.typing import NDArray


class BaseGraph(metaclass=ABCMeta):
    @property
    @abstractmethod
    def nodes(self) -> NodeView:
        pass

    @abstractmethod
    def adjacencies(self, u) -> List[Any]:
        pass

    @abstractmethod
    def has_adjacency(self, u, v) -> bool:
        pass

    @abstractmethod
    def add_node(self, node_for_adding, **attr):
        pass

    @abstractmethod
    def remove_node(self, u):
        pass


class MarkovianGraph(BaseGraph, metaclass=ABCMeta):
    @property
    @abstractmethod
    def edges(self) -> EdgeView:
        pass

    @abstractmethod
    def children(self, n) -> NodeView:
        pass

    @abstractmethod
    def parents(self, n) -> NodeView:
        pass

    @abstractmethod
    def add_edge(self, u_of_edge, v_of_edge, **attr):
        pass

    @abstractmethod
    def remove_edge(self, u, v):
        pass

    @abstractmethod
    def to_adjacency_graph(self) -> NDArray:
        pass

    @abstractmethod
    def compute_full_graph(self, to_networkx: bool = False) -> Union[nx.Graph, BaseGraph]:
        pass


class SemiMarkovianGraph(MarkovianGraph, metaclass=ABCMeta):
    @property
    @abstractmethod
    def bidirected_edges(self) -> EdgeView:
        pass

    @property
    @abstractmethod
    def c_components(self) -> List[Set]:
        pass

    @abstractmethod
    def has_bidirected_edge(self, u, v) -> bool:
        pass

    @abstractmethod
    def add_bidirected_edge(self, u_of_edge, v_of_edge, **attr):
        pass

    @abstractmethod
    def remove_bidirected_edge(self, u, v):
        pass


class MarkovEquivalenceClass(BaseGraph, metaclass=ABCMeta):
    @abstractmethod
    def possible_parents(self, n) -> NodeView:
        pass

    @abstractmethod
    def possible_children(self, n) -> NodeView:
        pass
