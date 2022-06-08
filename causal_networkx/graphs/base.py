from abc import ABCMeta, abstractmethod


class BaseGraph(metaclass=ABCMeta):
    @property
    @abstractmethod
    def nodes(self):
        return

    @abstractmethod
    def adjacencies(self, u):
        return

    @abstractmethod
    def has_adjacency(self, u, v):
        return

    @abstractmethod
    def add_node(self, node_for_adding, **attr):
        return

    @abstractmethod
    def remove_node(self, u):
        return


class MarkovianGraph(BaseGraph, metaclass=ABCMeta):
    @property
    @abstractmethod
    def edges(self):
        return

    @abstractmethod
    def children(self, n):
        return

    @abstractmethod
    def parents(self, n):
        return

    @abstractmethod
    def add_edge(self, u_of_edge, v_of_edge, **attr):
        return

    @abstractmethod
    def remove_edge(self, u, v):
        return

    @abstractmethod
    def to_adjacency_graph(self):
        return

    @abstractmethod
    def compute_full_graph(self, to_networkx: bool = False):
        return


class SemiMarkovianGraph(MarkovianGraph, metaclass=ABCMeta):
    @property
    @abstractmethod
    def bidirected_edges(self):
        return

    @property
    @abstractmethod
    def c_components(self):
        return

    @abstractmethod
    def has_bidirected_edge(self, u, v):
        return

    @abstractmethod
    def add_bidirected_edge(self, u_of_edge, v_of_edge, **attr):
        return

    @abstractmethod
    def remove_bidirected_edge(self, u, v):
        return


class MarkovEquivalenceClass(BaseGraph, metaclass=ABCMeta):
    @abstractmethod
    def possible_parents(self, n):
        return

    @abstractmethod
    def possible_children(self, n):
        return
