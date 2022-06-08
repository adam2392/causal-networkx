from typing import List, Set

import networkx as nx

from ..config import EdgeType
from .base import SemiMarkovianGraph
from .dag import DAG


# TODO: implement graph views for ADMG
class ADMG(DAG, SemiMarkovianGraph):
    """Acyclic directed mixed graph (ADMG).

    A causal graph with two different edge types: bidirected and traditional
    directed edges. Directed edges constitute causal relations as a
    causal DAG did, while bidirected edges constitute the presence of a
    latent confounder.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize directed edge graph. The edges in this graph
        represent directed edges between observed variables, which are
        represented using a ``networkx.DiGraph``, so accepts any arguments
        from the `networkx.DiGraph` class. There must be no cycles in this graph
        structure.

    incoming_latent_data : input graph (optional, default: None)
        Data to initialize bidirected edge graph. The edges in this graph
        represent bidirected edges, which are represented using a ``networkx.Graph``,
        so accepts any arguments from the `networkx.Graph` class.
    incoming_selection_bias : input graph (optional, default: None)
        Data to initialize selection bias graph. Currently,
        not used or implemented.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    networkx.DiGraph
    networkx.Graph
    DAG
    CPDAG
    PAG

    Notes
    -----
    The data structure underneath the hood is stored in two networkx graphs:
    ``networkx.Graph`` and ``networkx.DiGraph`` to represent the latent unobserved
    confounders and observed variables. These data structures should never be
    modified directly, but should use the ADMG class methods.

    - Bidirected edges (<->, indicating latent confounder) = networkx.Graph
    - Normal directed edges (<-, ->, indicating causal relationship) = networkx.DiGraph

    Nodes are defined as any nodes defined in the underlying ``DiGraph`` and
    ``Graph``. I.e. Any node connected with either a bidirected, or normal
    directed edge. Adding edges and bidirected edges are performed separately
    in different functions, compared to ``networkx``.

    Subclassing:
    All causal graphs are a mixture of graphs that represent the different
    types of edges possible. For example, a causal graph consists of two
    types of edges, directed, and bidirected. Each type of edge has the
    following operations:

    - has_<edge_type>_edge: Check if graph has this specific type of edge.
    - add_<edge_type>_edge: Add a specific edge type to the graph.
    - remove_<edge_type>_edge: Remove a specific edge type to the graph.

    All nodes are "stored" in ``self.dag``, which allows for isolated nodes
    that only have say bidirected edges pointing to it.
    """

    def __init__(
        self,
        incoming_graph_data=None,
        incoming_latent_data=None,
        incoming_selection_bias=None,
        **attr,
    ) -> None:
        # form the bidirected edge graph
        self.c_component_graph = nx.Graph(incoming_latent_data, **attr)

        # form selection bias graph
        # self.selection_bias_graph = nx.Graph(incoming_selection_bias, **attr)

        # call parent constructor
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

        # check that there is no cycles within the graph
        # self._edge_error_check()

    def _init_graphs(self):
        # create a list of the internal graphs
        self._graphs = [self.dag, self.c_component_graph]
        self._graph_names = [EdgeType.directed.value, EdgeType.bidirected.value]

        # number of edges allowed between nodes
        self.allowed_edges = 2

    @property
    def bidirected_edges(self):
        """Directed edges."""
        return self.c_component_graph.edges

    @property
    def c_components(self) -> List[Set]:
        """Generate confounded components of the graph.

        TODO: Improve runtime since this iterates over a list twice.

        Returns
        -------
        comp : list of set
            The c-components.
        """
        c_comps = nx.connected_components(self.c_component_graph)
        return [comp for comp in c_comps if len(comp) > 1]

    def _edge_error_check(self):
        if not nx.is_directed_acyclic_graph(self.dag):
            raise RuntimeError(f"{self.dag} is not a DAG, which it should be.")

    def number_of_bidirected_edges(self, u=None, v=None):
        """Return number of bidirected edges in graph."""
        return self.c_component_graph.number_of_edges(u=u, v=v)

    def has_bidirected_edge(self, u, v):
        """Check if graph has bidirected edge (u, v)."""
        if self.c_component_graph.has_edge(u, v):
            return True
        return False

    def __str__(self):
        return "".join(
            [
                type(self).__name__,
                f" named {self.name!r}" if self.name else "",
                f" with {self.number_of_nodes()} nodes, ",
                f"{self.number_of_edges()} edges and ",
                f"{self.number_of_bidirected_edges()} bidirected edges",
            ]
        )

    def compute_full_graph(self, to_networkx: bool = False):
        """Compute the full graph.

        Converts all bidirected edges to latent unobserved common causes.
        That is, if 'x <-> y', then it will transform to 'x <- [z] -> y'
        where [z] is "unobserved".

        TODO: add selection edges too

        Returns
        -------
        full_graph : nx.DiGraph
            The full graph.

        Notes
        -----
        The computation of the full graph is optimized by memoization of the
        hash of the edge list. When the hash does not change, it implies the
        edge list has not changed.

        Thus the conversion will not occur and the full graph will be read
        from memory.
        """
        from causal_networkx.utils import convert_latent_to_unobserved_confounders

        if self._current_hash != hash(self):
            explicit_G = convert_latent_to_unobserved_confounders(self)
            self._full_graph = explicit_G
            self._current_hash = hash(self)

        if to_networkx:
            return nx.DiGraph(self._full_graph.dag)  # type: ignore

        return self._full_graph

    def add_bidirected_edge(self, u_of_edge, v_of_edge, **attr) -> None:
        """Add a bidirected edge between u and v.

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
        self.c_component_graph.add_edge(u_of_edge, v_of_edge, **attr)

    def add_bidirected_edges_from(self, ebunch, **attr):
        """Add bidirected edges in a bunch."""
        self.c_component_graph.add_edges_from(ebunch, **attr)

    def remove_bidirected_edge(self, u_of_edge, v_of_edge, remove_isolate: bool = True) -> None:
        """Remove a bidirected edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        remove_isolate : bool
            Whether or not to remove isolated nodes after the removal
            of the bidirected edge. Default is True.

        See Also
        --------
        networkx.MultiDiGraph.add_edges_from : add a collection of edges
        networkx.MultiDiGraph.add_edge       : add an edge

        Notes
        -----
        ...
        """
        # add the bidirected arrow in
        self.c_component_graph.remove_edge(u_of_edge, v_of_edge)

        # remove nodes if they are isolated after removal of bidirected edges
        if remove_isolate:
            if u_of_edge in self.dag and nx.is_isolate(self.dag, u_of_edge):
                self.dag.remove_node(u_of_edge)
            if v_of_edge in self.dag and nx.is_isolate(self.dag, v_of_edge):
                self.dag.remove_node(v_of_edge)

    def do(self, nodes):
        """Apply a do-intervention on nodes to causal graph.

        Parameters
        ----------
        nodes : list of nodes | node
            Either a single node, or list of nodes.

        Returns
        -------
        causal_graph : ADMG
            The mutilated causal graph.

        Raises
        ------
        ValueError
            _description_
        """
        if not isinstance(nodes, list):
            nodes = [nodes]

        assert all(node in self.nodes for node in nodes)

        # create copies of total graph
        bd_graph = self.c_component_graph.copy()
        dag = self.dag.copy()

        for node in nodes:
            # remove any bidirected edges incident on nodes, which
            # results in removing the node from the bidirected graph
            bd_graph.remove_node(node)

            # remove any edges with parents into the nodes
            parent_dict = nx.predecessor(dag, node, cutoff=1)

            # remove the edge from parent -> node
            parents = parent_dict[node]
            for parent in parents:
                dag.remove_edge(parent, node)

        return ADMG(dag, bd_graph, **self.dag.graph)

    def soft_do(self, nodes, dependencies="original"):
        """Apply a soft-intervention on node to causal graph.

        Parameters
        ----------
        nodes : nodes
            A node within the graph.
        dependencies : list of nodes | str, optional
            What dependencies are now relevant for the node,
            by default 'original', which keeps all original
            directed edges (this still removes the bidirected
            edges). If a list of nodes, then it will add
            directed edges from those nodes to the node.

        Returns
        -------
        causal_graph : ADMG
            The mutilated graph.
        """
        # check that nodes and dependencies are same length
        if (not isinstance(dependencies, str)) and (not len(nodes) == len(dependencies)):
            raise ValueError(
                f"The number of nodes {len(nodes)} should match "
                f"the number of dependencies {len(dependencies)}."
            )

        assert all(node in self.nodes for node in nodes)

        # create copies of total graph
        bd_graph = self.c_component_graph.copy()
        dag = self.dag.copy()

        for idx, node in enumerate(nodes):
            # remove any bidirected edges incident on nodes, which
            # results in removing the node from the bidirected graph
            bd_graph.remove_node(node)

            if dependencies == "original":
                continue

            # remove any edges with parents into the nodes
            parent_dict = nx.predecessor(dag, node, cutoff=1)

            # remove the edge from parent -> node
            parents = parent_dict[node]
            for parent in parents:
                if parent not in dependencies[idx]:
                    dag.remove_edge(parent, node)

        return ADMG(dag, bd_graph, **self.dag.graph)

    def is_acyclic(self):
        """Check if graph is acyclic."""
        from causal_networkx.algorithms.dag import is_directed_acyclic_graph

        return is_directed_acyclic_graph(self)

    def subgraph(self, nodes):
        """Create a causal subgraph of just certain nodes."""
        pass

    def edge_subgraph(self, edges):
        """Create a causal subgraph of just certain edges."""
        pass

    def draw(self):
        """Draws causal graph.

        For custom parametrizations, use ``graphviz``
        or ``networkx`` drawers directly with the
        ``self.dag`` and ``self.c_component_graph``.
        """
        nx.draw_networkx(self.dag)
        nx.draw_networkx(self.c_component_graph, connectionstyle="arc3,rad=-0.4", style="dotted")

    def tomag(self):
        """Convert corresponding causal DAG to a MAG."""
        # add http://proceedings.mlr.press/v124/hu20a/hu20a.pdf algorithm
        pass

    def _classify_three_structure(self, a, b, c):
        """Classify three structure as a chain, fork or collider."""
        if self.dag.has_edge(a, b) and self.dag.has_edge(b, c):
            return "chain"

        if self.dag.has_edge(c, b) and self.dag.has_edge(b, a):
            return "chain"

        if self.dag.has_edge(a, b) and self.dag.has_edge(c, b):
            return "collider"

        if self.dag.has_edge(b, a) and self.dag.has_edge(b, c):
            return "fork"

        raise ValueError("Unsure how to classify ({},{},{})".format(a, b, c))

    def is_unshielded_collider(self, a, b, c):
        """Check if unshielded collider."""
        return self._classify_three_structure(a, b, c) == "collider" and not (
            self.has_edge(a, c) or self.has_edge(c, a)
        )
