from typing import Any, List, Optional

import networkx as nx

from ..config import EdgeType
from .mixins import AddingEdgeMixin, ExportMixin, GraphSampleMixin, NetworkXMixin


class DAG(NetworkXMixin, GraphSampleMixin, AddingEdgeMixin, ExportMixin):
    """Causal directed acyclic graph.

    This is a causal Bayesian network, or a Bayesian network
    with directed edges that constitute causal relations, rather than
    probabilistic dependences.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize directed edge graph. The edges in this graph
        represent directed edges between observed variables, which are
        represented using a ``networkx.DiGraph``, so accepts any arguments
        from the `networkx.DiGraph` class. There must be no cycles in this graph
        structure.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    networkx.DiGraph

    Notes
    -----
    ``_graphs`` and ``_graph_names`` private properties store graph
    objects denoting different types of edges and their corresponding
    names. These are useful for encoding various extensions of the causal DAG.
    """

    _graphs: List[nx.Graph]
    _graph_names: List[str]
    _current_hash: Optional[int]
    _full_graph: Optional[nx.DiGraph]

    def __init__(self, incoming_graph_data=None, **attr) -> None:
        # create the DAG of observed variables
        self.dag = nx.DiGraph(incoming_graph_data, **attr)

        # initialize the backend of graphs
        self._init_graphs()

        # keep track of the full graph
        self._full_graph = None
        self._current_hash = None
        if not self.is_acyclic():
            raise RuntimeError("Causal DAG must be acyclic.")

        # make sure to add all nodes to the dag that
        # are present in other internal graphs.
        # Note: This enables one to leverage the underlying DiGraph DAG
        # to do various graph traversals, such as d/m-separation.
        for graph in self._graphs:
            for node in graph.nodes:
                if node not in self:
                    self.dag.add_node(node)

    def _init_graphs(self):
        """Private function to initialize graphs.

        Should always be called after setting certain graph structures.
        """
        # create a list of the internal graphs
        self._graphs = [self.dag]
        self._graph_names = [EdgeType.directed.value]

        # number of edges allowed between nodes
        self.allowed_edges = 1

    def children(self, n):
        """Return an iterator over children of node n.

        Children of node 'n' are nodes with a directed
        edge from 'n' to that node. For example,
        'n' -> 'x', 'n' -> 'y'. Nodes only connected
        via a bidirected edge are not considered children:
        'n' <-> 'y'.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        children : Iterator
            An iterator of the children of node 'n'.
        """
        return self.dag.successors(n)

    def parents(self, n):
        """Return an iterator over parents of node n.

        Parents of node 'n' are nodes with a directed
        edge from 'n' to that node. For example,
        'n' <- 'x', 'n' <- 'y'. Nodes only connected
        via a bidirected edge are not considered parents:
        'n' <-> 'y'.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        parents : Iterator
            An iterator of the parents of node 'n'.
        """
        return self.dag.predecessors(n)

    def is_acyclic(self):
        """Check if graph is acyclic."""
        return nx.is_directed_acyclic_graph(self.dag)

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

        if raise_error:
            raise RuntimeError(
                f"There is already an existing edge between {u_of_edge} and {v_of_edge}. "
                f"Adding a {edge_type} edge is not possible. Please remove the existing "
                f"edge first."
            )

    def draw(self, **kwargs):
        nx.draw_networkx(self.dag, with_labels=True, **kwargs)

    def is_node_common_cause(self, node, exclude_nodes: List[Any] = None):
        """Check if a node is a common cause within the graph.

        Parameters
        ----------
        node : node
            A node in the graph.
        exclude_nodes : list, optional
            Set of nodes to exclude from consideration, by default None.

        Returns
        -------
        is_common_cause : bool
            Whether or not the node is a common cause or not.
        """
        if exclude_nodes is None:
            exclude_nodes = []

        successors = self.successors(node)
        count = 0
        for succ in successors:
            if succ not in exclude_nodes:
                count += 1
            if count >= 2:
                return True
        return False

    def set_nodes_as_latent_confounders(self, nodes):
        """Set nodes as latent unobserved confounders.

        Note that this only works if the original node is a common cause
        of some variables in the graph.

        Parameters
        ----------
        nodes : list
            A list of nodes to set. They must all be common causes of
            variables within the graph.

        Returns
        -------
        graph : ADMG
            The mixed-edge causal graph that results.
        """
        from causal_networkx.graphs.cgm import ADMG

        bidirected_edges = []
        new_graph = ADMG()

        for node in nodes:
            # check if the node is a common cause
            if not self.is_node_common_cause(node, exclude_nodes=nodes):
                raise RuntimeError(
                    f"{node} is not a common cause within the graph "
                    f"given excluding variables. This function will only convert common "
                    f"causes to latent confounders."
                )

            # keep track of which nodes to form c-components over
            successor_nodes = self.successors(node)
            for succ in successor_nodes:
                bidirected_edges.append((node, succ))

        # create the graph with nodes excluding those that are converted to latent confounders
        new_graph = ADMG(self.dag)
        new_graph.remove_nodes_from(nodes)

        # create the c-component structures
        new_graph.add_bidirected_edges_from(bidirected_edges)
        return new_graph
