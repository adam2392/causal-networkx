from collections import defaultdict
from typing import Any, Dict, FrozenSet

import networkx as nx

from ..config import EdgeType
from .base import MarkovEquivalenceClass
from .dag import DAG


class CPDAG(DAG, MarkovEquivalenceClass):
    """Completed partially directed acyclic graphs (CPDAG).

    CPDAGs generalize causal DAGs by allowing undirected edges.
    Undirected edges imply uncertainty in the orientation of the causal
    relationship. For example, ``A - B``, can be ``A -> B`` or ``A <- B``,
    allowing for a Markov equivalence class of DAGs for each CPDAG.
    This means that the number of DAGs represented by a CPDAG is $2^n$, where
    ``n`` is the number of undirected edges present in the CPDAG.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize directed edge graph. The edges in this graph
        represent directed edges between observed variables, which are
        represented using a ``networkx.DiGraph``, so accepts any arguments
        from the `networkx.DiGraph` class. There must be no cycles in this graph
        structure.

    incoming_uncertain_data : input graph (optional, default: None)
        Data to initialize undirected edge graph. The edges in this graph
        represent undirected edges, which are represented using a ``networkx.Graph``,
        so accepts any arguments from the `networkx.Graph` class.

    See Also
    --------
    networkx.DiGraph
    networkx.Graph
    DAG
    ADMG
    PAG
    causal_networkx.discovery.PC

    Notes
    -----
    CPDAGs are Markov equivalence class of causal DAGs. The implicit assumption in
    these causal graphs are the Structural Causal Model (or SCM) is Markovian, inducing
    causal sufficiency, where there is no unobserved latent confounder. This allows CPDAGs
    to be learned from score-based (such as the GES algorithm) and constraint-based
    (such as the PC algorithm) approaches for causal structure learning.

    One should not use CPDAGs if they suspect their data has unobserved latent confounders.
    """

    def __init__(self, incoming_graph_data=None, incoming_uncertain_data=None, **attr) -> None:
        self.undirected_edge_graph = nx.Graph(incoming_uncertain_data, **attr)

        super().__init__(incoming_graph_data, **attr)
        self._check_cpdag()
        self._excluded_triples = defaultdict(frozenset)

    def _init_graphs(self):
        # create a list of the internal graphs
        self._graphs = [self.dag, self.undirected_edge_graph]
        self._graph_names = [EdgeType.directed.value, EdgeType.undirected.value]

        # number of edges allowed between nodes
        self.allowed_edges = 1

    def __str__(self):
        return "".join(
            [
                type(self).__name__,
                f" named {self.name!r}" if self.name else "",
                f" with {self.number_of_nodes()} nodes, ",
                f"{self.number_of_edges()} edges and ",
                f"{self.number_of_undirected_edges()} undirected edges",
            ]
        )

    @property
    def undirected_edges(self):
        """Return the undirected edges of the graph."""
        return self.undirected_edge_graph.edges

    def _check_cpdag(self):
        """Check for errors in the CPDAG construction.

        Checks if there is a violation of the DAG property.
        """
        if not self.is_acyclic():
            raise RuntimeError("DAG constructed was not acyclic.")

    def add_undirected_edge(self, u_of_edge, v_of_edge, **attr) -> None:
        """Add a undirected edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph. Moreover, they will be added
        to the underlying DiGraph, which stores all regular
        directed edges.

        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        See Also
        --------
        networkx.Graph.add_edges_from : add a collection of edges
        networkx.Graph.add_edge       : add an edge

        Notes
        -----
        ...
        """
        # if the nodes connected are not in the dag, then
        # add them into the observed variable graph
        if u_of_edge not in self.dag:
            self.dag.add_node(u_of_edge)
        if v_of_edge not in self.dag:
            self.dag.add_node(v_of_edge)

        # add the bidirected arrow in
        self.undirected_edge_graph.add_edge(u_of_edge, v_of_edge, **attr)

    def add_undirected_edges_from(self, ebunch, **attr):
        """Add undirected edges in a bunch."""
        self.undirected_edge_graph.add_edges_from(ebunch, **attr)

    def remove_undirected_edge(self, u, v):
        """Remove circle edge from graph."""
        self.undirected_edge_graph.remove_edge(u, v)

    def _check_adding_edge(self, u_of_edge, v_of_edge, edge_type):
        """Check compatibility among internal graphs when adding an edge of a certain type.

        Parameters
        ----------
        u_of_edge : node
            The start node.
        v_of_edge : node
            The end node.
        edge_type : str of EdgeType
            The edge type that is being added.
        """
        raise_error = False
        if edge_type == EdgeType.directed.value:
            # there should not be a circle edge, or a bidirected edge
            if u_of_edge == v_of_edge:
                raise_error = True
            if self.has_edge(v_of_edge, u_of_edge):
                raise RuntimeError(
                    f"There is an existing {v_of_edge} -> {u_of_edge}. You are "
                    f"trying to add a directed edge from {u_of_edge} -> {v_of_edge}. "
                    f"If your intention is to create a bidirected edge, first remove the "
                    f"edge and then explicitly add the bidirected edge."
                )
        elif edge_type == EdgeType.undirected.value:
            # there should not be any type of edge between the two
            if self.has_adjacency(u_of_edge, v_of_edge):
                raise_error = True

        if raise_error:
            raise RuntimeError(
                f"There is already an existing edge between {u_of_edge} and {v_of_edge}. "
                f"Adding a {edge_type} edge is not possible. Please remove the existing "
                f"edge first."
            )

    def number_of_undirected_edges(self, u=None, v=None):
        """Return number of undirected edges in graph."""
        return self.undirected_edge_graph.number_of_edges(u=u, v=v)

    def has_undirected_edge(self, u, v):
        """Check if graph has undirected edge (u, v)."""
        if self.undirected_edge_graph.has_edge(u, v):
            return True
        return False

    def draw(self):
        """Draws CPDAG graph.

        For custom parametrizations, use ``graphviz``
        or ``networkx`` drawers directly with the
        ``self.dag`` and ``self.c_component_graph``.
        """
        nx.draw_networkx(self.dag)
        nx.draw_networkx(self.undirected_edge_graph)

    def orient_undirected_edge(self, u, v):
        """Orient undirected edge into an arrowhead.

        If there is an undirected edge u - v, then the arrowhead
        will orient u -> v. If the correct order is v <- u, then
        simply pass the arguments in different order.

        Parameters
        ----------
        u : node
            The parent node
        v : node
            The node that 'u' points to in the graph.
        """
        if not self.has_undirected_edge(u, v):
            raise RuntimeError(f"There is no undirected edge between {u} and {v}.")

        self.remove_undirected_edge(v, u)
        self.add_edge(u, v)

    def to_directed(self):
        """Convert CPDAG to a networkx DiGraph.

        Undirected edges are converted to a '->' and '<-' edge in the DiGraph.
        Note that the resulting DiGraph is not "acyclic" anymore.
        """
        graph = self.dag.copy()
        ud_graph = self.undirected_edge_graph.to_directed()
        return nx.compose(graph, ud_graph)

    def has_uncertain_edge(self, u, v):
        """Check if graph has undirected edge (u, v)."""
        raise NotImplementedError()

    def possible_children(self, n):
        pass

    def possible_parents(self, n):
        pass

    @property
    def excluded_triples(self) -> Dict[FrozenSet, Any]:
        return self._excluded_triples


class ExtendedPattern(CPDAG):
    def __init__(self, incoming_graph_data=None, incoming_uncertain_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, incoming_uncertain_data, **attr)
        self._unfaithful_triples = dict()

    def mark_unfaithful_triple(self, v_i, u, v_j):
        if any(node not in self.nodes for node in [v_i, u, v_j]):
            raise RuntimeError(f"The triple {v_i}, {u}, {v_j} is not in the graph.")

        self._unfaithful_triples[frozenset(v_i, u, v_j)] = None

    @property
    def unfaithful_triples(self):
        return self._unfaithful_triples

    @property
    def excluded_triples(self) -> Dict:
        excluded_trips = super().excluded_triples().copy()
        excluded_trips |= self.unfaithful_triples
        return excluded_trips
