from __future__ import annotations

import typing
from typing import List, Optional, Protocol, Set

import networkx as nx
import pandas as pd
from networkx import NetworkXError
from numpy.typing import NDArray

from causal_networkx.config import PAG_EDGE_MAPPING, EdgeType


class NetworkXMixin(Protocol):
    """Suite of overridden methods of Networkx Graphs.

    Assumes the subclass will store a DiGraph in a class
    attribute called 'dag' and also has an arbitrary list
    of graphs in an attribute '_graphs', which can be
    used to query basic information.

    These methods and properties override what is expected behavior for
    causal graphs.
    """

    @property
    def name(self):
        """Name as a string identifier of the graph.

        This graph attribute appears in the attribute dict G.graph
        keyed by the string "name". as well as an attribute (technically
        a property) ``G.name``. This is entirely user controlled.
        """
        return self.dag.name

    @name.setter
    def name(self, s):
        """Set the name of the graph."""
        self.dag["name"] = s

    def predecessors(self, u):
        """Return predecessors of node u.

        A predecessor is defined as nodes with a directed edge to 'u'.
        That is 'v' -> 'u'. A bidirected edge would not qualify as a
        predecessor.
        """
        return self.dag.predecessors(u)

    def successors(self, u):
        """Return successors of node u.

        A successor is defined as nodes with a directed edge from 'u'.
        That is 'u' -> 'v'. A bidirected edge would not qualify as a
        successor.
        """
        return self.dag.successors(u)

    def get_edge_data(self, u, v, default=None):
        """Get edge data from underlying DiGraph."""
        return self.dag.get_edge_data(u, v, default)

    def clear_edges(self):
        """Remove all edges from causal graph without removing nodes.

        Clears edges in the DiGraph and the bidirected undirected graph.
        """
        for graph in self._graphs:
            graph.clear_edges()

    def clear(self):
        """Remove all nodes and edges in graphs."""
        for graph in self._graphs:
            graph.clear()

    @property
    def edges(self):
        """Directed edges."""
        return self.dag.edges

    def number_of_edges(self):
        """Return number of edges in graph."""
        return len(self.edges)

    @property
    def nodes(self, data=False):
        """Return the nodes within the DAG.

        Ignores the c-component graph nodes.
        """
        return self.dag.nodes(data=data)  # ).union(set(self.c_component_graph.nodes))

    def has_adjacency(self, u, v):
        """Check if there is any edge between u and v."""
        if any(graph.has_edge(u, v) or graph.has_edge(v, u) for graph in self._graphs):
            return True
        return False

    def adjacencies(self, u):
        """Get all adjacent nodes to u.

        Adjacencies are defined as any type of edge to node 'u'.
        """
        nghbrs = dict()
        for graph in self._graphs:
            graph = graph.to_undirected()
            if u in graph:
                nghbrs.update({node: None for node in graph.neighbors(u)})
        return list(nghbrs.keys())

    def number_of_nodes(self):
        """Return number of nodes in graph."""
        return len(self.nodes)

    def number_of_edges(self):
        """Return number of edges in graph."""
        return len(self.edges)

    def has_node(self, n):
        """Check if graph has node 'n'."""
        return n in self

    def adjacencies(self, u):
        """Get all adjacent nodes to u.

        Adjacencies are defined as any type of edge to node 'u'.
        """
        nghbrs = dict()
        for graph in self._graphs:
            graph = graph.to_undirected()
            if u in graph:
                nghbrs.update({node: None for node in graph.neighbors(u)})
        return list(nghbrs.keys())

    def __contains__(self, n):
        """Return True if n is a node, False otherwise. Use: 'n in G'.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> 1 in G
        True
        """
        try:
            return n in self.nodes
        except TypeError:
            return False

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, n):
        return self.dag[n]

    def __hash__(self) -> int:
        all_edges = []
        for graph in self._graphs:  # type: ignore
            all_edges.extend(graph.edges)
        return hash(tuple(all_edges))

    def add_node(self, node_for_adding, **attr):
        """Add node to causal graph."""
        self.dag.add_node(node_for_adding=node_for_adding, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add nodes to causal graph."""
        self.dag.add_nodes_from(nodes_for_adding, **attr)

    def remove_node(self, n):
        """Remove node in causal graphs."""
        for graph in self._graphs:
            try:
                graph.remove_node(n)
            except NetworkXError as e:
                if isinstance(graph, nx.DiGraph):
                    raise (e)

    def remove_nodes_from(self, ebunch):
        """Remove nodes from causal graph."""
        for graph in self._graphs:
            try:
                graph.remove_nodes_from(ebunch)
            except NetworkXError as e:
                if isinstance(graph, nx.DiGraph):
                    raise (e)

    def has_edge(self, u, v):
        """Check if graph has edge (u, v)."""
        return self.dag.has_edge(u, v)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

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
        add_edges_from : add a collection of edges

        Notes
        -----
        Adding an edge that already exists updates the edge data.

        Many NetworkX algorithms designed for weighted graphs use
        an edge attribute (by default ``weight``) to hold a numerical value.

        Examples
        --------
        The following all add the edge e=(1, 2) to graph G:

        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = (1, 2)
        >>> G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container

        Associate data to edges using keywords:

        >>> G.add_edge(1, 2, weight=3)
        >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        For non-string attribute keys, use subscript notation.

        >>> G.add_edge(1, 2)
        >>> G[1][2].update({0: 5})
        >>> G.edges[1, 2].update({0: 5})
        """
        self.dag.add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch, **attr):
        """Add directed edges."""
        self.dag.add_edges_from(ebunch, **attr)

    def remove_edges_from(self, ebunch):
        """Remove directed edges."""
        self.dag.remove_edges_from(ebunch)

    def remove_edge(self, u, v):
        """Remove directed edge."""
        self.dag.remove_edge(u, v)

    def copy(self):
        """Return a copy of the causal graph."""
        return self.__class__(*self._graphs, **self.dag.graph)

    def order(self):
        """Return the order of the DiGraph."""
        return self.dag.order()

    def size(self, weight=None):
        """Return the total number of edges possibly with weights."""
        size_ = 0
        for graph in self._graphs:
            size_ += graph.size(weight)
        return size_

    def degree(self, n):
        """Compute the degree of the DiGraph."""
        return self.dag.degree(n)


class GraphSampleMixin:
    def dummy_sample(self):
        """Sample an empty dataframe with columns as the nodes.

        Used for oracle testing.
        """
        df_values = dict()
        for node in self.nodes:
            df_values[node] = []

        df = pd.DataFrame.from_dict(df_values)
        return df

    def sample(self, n=1000):
        """Sample from a graph."""
        df_values = []

        # construct truth-table based on the SCM
        for _ in range(n):
            # sample now all observed variables
            for endog, endog_func in self.endogenous.items():
                endog_value = self._sample_function(endog_func, self.symbolic_runtime)

                if endog not in self.symbolic_runtime:
                    self.symbolic_runtime[endog] = endog_value

            # add each sample to
            df_values.append(self.symbolic_runtime)

        # now convert the final sample to a dataframe
        # result_df = pd.DataFrame(df_values)

        # if not include_latents:
        #     # remove latent variable columns
        #     result_df.drop(self.exogenous.keys(), axis=1, inplace=True)
        # else:
        #     # make sure to order the columns with latents first
        #     def key(x):
        #         return x not in self.exogenous.keys()

        #     result_df = result_df[sorted(result_df, key=key)]
        return df_values


class AddingEdgeMixin:
    def add_chain(self, node_chain):
        """Add a causal chain."""
        ebunch = []
        for idx, node in enumerate(node_chain[:-1]):
            ebunch.append((node, node_chain[idx + 1]))
        self.add_edges_from(ebunch)


class ExportMixin:
    """Mixin class for exporting causal graphs to other formats."""

    def to_dot_graph(self, to_dagitty: bool = False) -> str:
        """Convert to 'dot' graph representation.

        The DOT language for graphviz is what is commonly
        used in R's ``dagitty`` package. This is a string
        representation of the graph. However, this converts to a
        string format that is not 100% representative of DOT. See
        Notes for more information.

        Parameters
        ----------
        to_dagitty : bool
            Whether to conform to the Dagitty format, where the
            string begins with ``dag {`` instead of ``strict digraph {``.

        Returns
        -------
        dot_graph : str
            A string representation in DOT format for the graph.

        Notes
        -----
        The output of this function can be immediately plugged into
        the dagitty online portal for drawing a graph.

        For example, if we have a mixed edge graph, with directed
        and bidirected arrows (i.e. a causal DAG). Specifically, if
        we had ``0 -> 1`` with a latent confounder, we would get the
        following output:

            strict digraph {
                0;
                1;
                0 -> 1;
                0 <-> 1;
            }

        To represent for example a bidirected edge, ``A <-> B``,
        the DOT format would make you use ``A -> B [dir=both]``, but
        this is not as intuitive. ``A <-> B`` also complies with dagitty
        and other approaches to drawing graphs in Python/R.

        Reference
        ---------
        https://github.com/pydot/pydot
        """
        node_str_list = []
        for node in self.nodes:
            node_str_list.append(f"{node};")
        node_str = "\n".join(node_str_list)

        # for each graph handle edges' string representation
        # differently
        edge_str_list = []
        for name, graph in zip(self._graph_names, self._graphs):
            dot = nx.nx_pydot.to_pydot(graph)
            dot_str = dot.to_string()

            # only keep rows with edge strings
            edge_list = []
            for row in dot_str.split("\n")[1:-2]:
                if f"{row}" not in node_str_list:
                    # replace edge marks with their appropriate string representation
                    if name == EdgeType.bidirected.value:
                        row = row.replace("--", "<->")
                    elif name == EdgeType.circle.value:
                        row = row.replace("->", "-o")
                    edge_list.append(row)
            edge_str_list.extend(edge_list)
        edge_str = "\n".join(edge_str_list)

        # form the final DOT string representation
        if to_dagitty:
            header = "dag"
        else:
            header = "strict digraph"
        dot_graph = header + " {\n" f"{node_str}\n" f"{edge_str}\n" "}"
        return dot_graph


class DAG(NetworkXMixin, GraphSampleMixin, AddingEdgeMixin, ExportMixin):
    """Causal directed acyclic graph.

    This is a causal Bayesian network, or a Bayesian network
    with directed edges that constitute causal relations, rather than
    probabilistic dependences.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize directed acyclic graph. If None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.  If the corresponding optional Python
        packages are installed the data can also be a 2D NumPy array, a
        SciPy sparse matrix, or a PyGraphviz graph. The edges in this graph
        represent directed edges between observed variables, which are
        represented using a ``networkx.DiGraph``. This is a DAG, meaning
        there are no cycles.

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

    def _init_graphs(self):
        """Private function to initialize graphs.

        Should always be called after setting certain graph structures.
        """
        # create a list of the internal graphs
        self._graphs = [self.dag]
        self._graph_names = [EdgeType.arrow.value]

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

    def has_edge(self, u, v):
        """Check if graph has edge (u, v)."""
        return self.dag.has_edge(u, v)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

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
        add_edges_from : add a collection of edges

        Notes
        -----
        Adding an edge that already exists updates the edge data.

        Many NetworkX algorithms designed for weighted graphs use
        an edge attribute (by default ``weight``) to hold a numerical value.

        Examples
        --------
        The following all add the edge e=(1, 2) to graph G:

        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = (1, 2)
        >>> G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container

        Associate data to edges using keywords:

        >>> G.add_edge(1, 2, weight=3)
        >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        For non-string attribute keys, use subscript notation.

        >>> G.add_edge(1, 2)
        >>> G[1][2].update({0: 5})
        >>> G.edges[1, 2].update({0: 5})
        """
        self.dag.add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch, **attr):
        """Add directed edges."""
        self.dag.add_edges_from(ebunch, **attr)

    def remove_edges_from(self, ebunch):
        """Remove directed edges."""
        self.dag.remove_edges_from(ebunch)

    def remove_edge(self, u, v):
        """Remove directed edge."""
        self.dag.remove_edge(u, v)

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
        if edge_type == EdgeType.arrow.value:
            # there should not be a circle edge, or a bidirected edge
            if u_of_edge == v_of_edge:
                raise_error = True

        if raise_error:
            raise RuntimeError(
                f"There is already an existing edge between {u_of_edge} and {v_of_edge}. "
                f"Adding a {edge_type} edge is not possible. Please remove the existing "
                f"edge first."
            )


class CPDAG(DAG):
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
        Data to initialize directed acyclic graph. If None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.  If the corresponding optional Python
        packages are installed the data can also be a 2D NumPy array, a
        SciPy sparse matrix, or a PyGraphviz graph. The edges in this graph
        represent directed edges between observed variables, which are
        represented using a ``networkx.DiGraph``. This is a DAG, meaning
        there are no cycles.
    incoming_uncertain_data : input graph (optional, default: None)
        Data to initialize bidirected edge graph. If None (default) an empty
        graph is created. The data format can be any as ``incoming_graph_data``.
        The edges in this graph represent bidirected edges, which are
        represented using a ``networkx.Graph``.

    See Also
    --------
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

    def _init_graphs(self):
        # create a list of the internal graphs
        self._graphs = [self.dag, self.undirected_edge_graph]
        self._graph_names = [EdgeType.arrow.value, EdgeType.undirected.value]

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
        if edge_type == EdgeType.arrow.value:
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

    def all_edges(self):
        """Get dictionary of all the edges by edge type."""
        return {
            "edges": self.edges,
            "undirected": self.undirected_edges,
        }

    def to_directed(self):
        """Convert CPDAG to a networkx DiGraph.

        Undirected edges are converted to a '->' and '<-' edge in the DiGraph.
        Note that the resulting DiGraph is not "acyclic" anymore.
        """
        graph = self.dag.copy()
        ud_graph = self.undirected_edge_graph.to_directed()
        return nx.compose(graph, ud_graph)


# TODO: implement graph views for ADMG
class ADMG(DAG):
    """Acyclic directed mixed graph (ADMG).

    A causal graph with two different edge types: bidirected and traditional
    directed edges. Directed edges constitute causal relations as a
    causal DAG did, while bidirected edges constitute the presence of a
    latent confounder.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize directed acyclic graph. If None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.  If the corresponding optional Python
        packages are installed the data can also be a 2D NumPy array, a
        SciPy sparse matrix, or a PyGraphviz graph. The edges in this graph
        represent directed edges between observed variables, which are
        represented using a ``networkx.DiGraph``. This is a DAG, meaning
        there are no cycles.

    incoming_latent_data : input graph (optional, default: None)
        Data to initialize bidirected edge graph. If None (default) an empty
        graph is created. The data format can be any as ``incoming_graph_data``.
        The edges in this graph represent bidirected edges, which are
        represented using a ``networkx.Graph``.

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

    _cond_set: Set

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

        # the conditioning set used in d-separation
        # keep track of variables that are always conditioned on
        self._cond_set = set()

        # check that there is no cycles within the graph
        # self._edge_error_check()

    def _init_graphs(self):
        # create a list of the internal graphs
        self._graphs = [self.dag, self.c_component_graph]
        self._graph_names = [EdgeType.arrow.value, EdgeType.bidirected.value]

        # number of edges allowed between nodes
        self.allowed_edges = 2

    def to_adjacency_graph(self):
        """Compute an adjacency undirected graph.

        Two nodes are considered adjacent if there exist
        any type of edge between the two nodes.
        """
        # form the undirected graph of all inner graphs
        graph_list = []
        for graph in self._graphs:
            graph_list.append(graph.to_undirected())

        adj_graph = graph_list[0]
        for idx in range(1, len(graph_list)):
            adj_graph = nx.compose(adj_graph, graph_list[idx])
        return adj_graph

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


# TODO: implement m-separation algorithm
class PAG(ADMG):
    """Partial ancestral graph (PAG).

    An equivalence class of MAGs, which represent a large
    equivalence class then of causal DAGs.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize directed edge graph. If None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.  If the corresponding optional Python
        packages are installed the data can also be a 2D NumPy array, a
        SciPy sparse matrix, or a PyGraphviz graph. The edges in this graph
        represent directed edges between observed variables, which are
        represented using a ``networkx.DiGraph``. This is a DAG, meaning
        there are no cycles.

    incoming_latent_data : input graph (optional, default: None)
        Data to initialize bidirected edge graph. If None (default) an empty
        graph is created. The data format can be any as ``incoming_graph_data``.
        The edges in this graph represent bidirected edges, which are
        represented using a ``networkx.Graph``.

    incoming_uncertain_data : input graph (optional, default: None)
        Data to initialize circle edge graph. If None (default) an empty
        graph is created. The data format can be any as ``incoming_graph_data``.
        The edges in this graph represent circle edges, which are represented
        using a ``networkx.DiGraph``. This does not necessarily need to be a DAG,
        since there are circle edges possibly in both directions.

    incoming_selection_bias : input graph (optional, default: None)
        Data to initialize selection bias graph. Currently,
        not used or implemented.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.


    Notes
    -----
    In PAGs, there is only one edge between any two nodes, but there are
    different types of edges. The entire PAG is represented using multiple
    ``networkx`` graphs. Together, these graphs are joined together to form an efficient
    representation of the PAG.

    - directed edges (->, <-, indicating causal relationship): ``networkx.DiGraph``
    - bidirected edges (<->, indicating latent confounder): ``networkx.DiGraph``
    - circular edges (-o, o-, indicating uncertainty in edge type): ``networkx.DiGraph``
    - undirected edges (-, indicating selection bias): ``networkx.Graph``. Currently
      not implemented or used.

    Compared to causal graphs, PAGs differ in terms of how parents and children
    are defined. In causal graphs, there are only two types of edges, either a
    directed arrow, or bidirected arrow. In PAGs, there are directed arrows with
    either a circle, or tail on the other end: 'x' o-> 'y'. This now introduces
    "possible" parents/children denoted with a circle edge on the other end of the
    arrow and definite parents/children with only an arrow edge.

    Since PAGs only allow "one edge" between any two nodes, adding edges and removing
    edges have different semantics. See more in `add_edge`, `remove_edge`.
    """

    def __init__(
        self,
        incoming_graph_data=None,
        incoming_latent_data=None,
        incoming_uncertain_data=None,
        incoming_selection_data=None,
        **attr,
    ) -> None:
        self.circle_edge_graph = nx.DiGraph(incoming_uncertain_data, **attr)

        # construct the causal graph
        super().__init__(
            incoming_graph_data=incoming_graph_data,
            incoming_latent_data=incoming_latent_data,
            incoming_selection_bias=incoming_selection_data,
            **attr,
        )

        # check the PAG
        self._check_pag()

    def all_edges(self):
        """Get dictionary of all the edges by edge type."""
        return {
            "edges": self.edges,
            "bidirected": self.bidirected_edges,
            "circle": self.circle_edges,
        }

    def _init_graphs(self):
        self._graphs = [
            self.dag,
            self.c_component_graph,
            self.circle_edge_graph,
        ]
        self._graph_names = [EdgeType.arrow.value, EdgeType.bidirected.value, EdgeType.circle.value]

        # number of allowed edges between any two nodes
        self.allowed_edges = 1

    def _check_circle_edge(self, node, nghbr):
        raise_error = False
        # check that there is no bidirected arrow in either direction
        if self.has_bidirected_edge(node, nghbr) or self.has_bidirected_edge(nghbr, node):
            raise_error = True

        # check that there is no directed arrow in node -> ngbhr
        elif self.has_edge(node, nghbr):
            raise_error = True

        # check if there is a circle edge in the other direction
        elif self.has_circle_edge(nghbr, node):
            # if so, check that no arrow is also in that direction
            if self.has_edge(nghbr, node):
                raise_error = True

        if raise_error:
            raise RuntimeError(
                f"There are multiple edges between {node} and {nghbr} already "
                f"in {self} when trying to add a circle edge."
            )

    def _check_arrow_edge(self, node, nghbr):
        raise_error = False
        # check that there is no bidirected arrow in either direction
        if self.has_bidirected_edge(node, nghbr):
            raise_error = True

        # check that there is no directed circle edge in node -o ngbhr
        elif self.has_circle_edge(node, nghbr):
            raise_error = True

        # check if there is an arrow edge in the other direction
        elif self.has_edge(nghbr, node):
            raise_error = True

        if raise_error:
            raise RuntimeError(
                f"There are multiple edges between {node} and {nghbr} already "
                f"in {self} when trying to add an arrow edge."
            )

    def _check_bidirected_edge(self, node, nghbr):
        raise_error = False
        # can't have any neighbor to node, or vice versa
        # check that there is no directed circle edge in node -o ngbhr
        if self.has_circle_edge(node, nghbr) or self.has_circle_edge(nghbr, node):
            raise_error = True

        # check if there is an arrow edge in the other direction
        elif self.has_edge(nghbr, node) or self.has_edge(node, nghbr):
            raise_error = True

        if raise_error:
            raise RuntimeError(
                f"There are multiple edges between {node} and {nghbr} already "
                f"in {self} when trying to add a bidirected edge."
            )

    def _check_pag(self):
        """Check for errors in the PAG construction.

        Checks if there are multiple edges between any two nodes,
        which is a violation of the PAG definition.

        Since this iterates over all edges of the PAG, this should
        only be called once during construction of the PAG. Otherwise
        adding edges via the PAG API does quick checks on the graph
        to ensure it does not violate the number of allowed edges
        between any two nodes.
        """
        for node in self.nodes:
            # get neighbors that are adjacent with any edge
            for nghbr in self.neighbors(node):
                if self.edge_type(node, nghbr) == "circle":
                    self._check_circle_edge(node, nghbr)
                elif self.edge_type(node, nghbr) == "arrow":
                    # check similar qualities
                    self._check_arrow_edge(node, nghbr)
                elif self.edge_type(node, nghbr) == "bidirected":
                    # then this is a bidirected edge
                    self._check_bidirected_edge(node, nghbr)

    def __str__(self):
        return "".join(
            [
                type(self).__name__,
                f" named {self.name!r}" if self.name else "",
                f" with {self.number_of_nodes()} nodes, ",
                f"{self.number_of_edges()} edges, ",
                f"{self.number_of_bidirected_edges()} bidirected edges and ",
                f"{self.number_of_circle_edges()} circle edges.",
            ]
        )

    def possible_parents(self, n):
        """Return the possible parents of node 'n' in a PAG.

        Possible parents of 'n' are nodes with an edge like
        'n' <-o 'x'. Nodes with 'n' o-o 'x' are not considered
        possible parents.

        Parameters
        ----------
        n : node
            A node in the PAG.

        Returns
        -------
        parents : Iterator
            An iterator of the parents of node 'n'.

        See Also
        --------
        possible_children
        parents
        children
        """
        return super().parents(n)

    def possible_children(self, n):
        """Return the possible children of node 'n' in a PAG.

        Possible children of 'n' are nodes with an edge like
        'n' o-> 'x'. Nodes with 'n' o-o 'x' are not considered
        possible children.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        children : Iterator
            An iterator of the children of node 'n'.

        See Also
        --------
        children
        parents
        possible_parents
        """
        return super().children(n)

    def parents(self, n):
        """Return the definite parents of node 'n' in a PAG.

        Definite parents are parents of node 'n' with only
        a directed edge between them from 'n' <- 'x'. For example,
        'n' <-o 'x' does not qualify 'x' as a parent of 'n'.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Yields
        ------
        parents : Iterator
            An iterator of the definite parents of node 'n'.

        See Also
        --------
        possible_children
        children
        possible_parents
        """
        possible_parents = self.possible_parents(n)
        for node in possible_parents:
            if not self.has_circle_edge(n, node):
                yield node

    def children(self, n):
        """Return the definite children of node 'n' in a PAG.

        Definite children are children of node 'n' with only
        a directed edge between them from 'n' -> 'x'. For example,
        'n' o-> 'x' does not qualify 'x' as a children of 'n'.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Yields
        ------
        children : Iterator
            An iterator of the children of node 'n'.

        See Also
        --------
        possible_children
        parents
        possible_parents
        """
        possible_children = self.possible_children(n)
        for node in possible_children:
            if not self.has_circle_edge(node, n):
                yield node

    @property
    def circle_edges(self):
        """Return all circle edges."""
        return self.circle_edge_graph.edges

    def number_of_circle_edges(self, u=None, v=None):
        """Return number of bidirected edges in graph."""
        return self.circle_edge_graph.number_of_edges(u=u, v=v)

    def _check_adding_edges(self, ebunch, edge_type):
        """Check adding edges as a bunch.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the
            graph. The edges must be given as 2-tuples (u, v) or
            3-tuples (u, v, d) where d is a dictionary containing edge data.
        edge_type : str of EdgeType
            The edge type that is being added.
        """
        for e in ebunch:
            if len(e) == 3:
                raise NotImplementedError("Adding edges with data is not supported yet.")
            u, v = e
            self._check_adding_edge(u, v, edge_type)

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
        if edge_type == EdgeType.circle.value:
            # there should not be an existing arrow
            # nor a bidirected arrow
            if self.has_edge(u_of_edge, v_of_edge) or self.has_bidirected_edge(
                u_of_edge, v_of_edge
            ):
                raise_error = True
        elif edge_type == EdgeType.arrow.value:
            # there should not be a circle edge, or a bidirected edge
            if self.has_circle_edge(u_of_edge, v_of_edge) or self.has_bidirected_edge(
                u_of_edge, v_of_edge
            ):
                raise_error = True
            if self.has_edge(v_of_edge, u_of_edge):
                raise RuntimeError(
                    f"There is an existing {v_of_edge} -> {u_of_edge}. You are "
                    f"trying to add a directed edge from {u_of_edge} -> {v_of_edge}. "
                    f"If your intention is to create a bidirected edge, first remove the "
                    f"edge and then explicitly add the bidirected edge."
                )
        elif edge_type == EdgeType.bidirected.value:
            # there should not be any type of edge between the two
            if self.has_adjacency(u_of_edge, v_of_edge):
                raise_error = True

        if raise_error:
            raise RuntimeError(
                f"There is already an existing edge between {u_of_edge} and {v_of_edge}. "
                f"Adding a {edge_type} edge is not possible. Please remove the existing "
                f"edge first."
            )

    def add_circle_edge(self, u_of_edge, v_of_edge, bidirected: bool = False):
        """Add a circle edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        bidirected : bool
            Whether or not to also add an uncertain edge
            from ``v_of_edge`` to ``u_of_edge``.

        See Also
        --------
        add_edges_from : add a collection of edges
        add_edge

        """
        self._check_adding_edge(u_of_edge, v_of_edge, EdgeType.circle.value)

        if not self.dag.has_edge(v_of_edge, u_of_edge) and not bidirected:
            raise RuntimeError(
                f"There is no directed edge from {v_of_edge} to {u_of_edge}. "
                f"Adding a single circle edge is redundant. Are you sure you "
                f"do not intend on adding a bidrected edge?"
            )
        # if the nodes connected are not in the dag, then
        # add them into the observed variable graph
        if u_of_edge not in self.dag:
            self.dag.add_node(u_of_edge)
        if v_of_edge not in self.dag:
            self.dag.add_node(v_of_edge)

        self.circle_edge_graph.add_edge(u_of_edge, v_of_edge)
        if bidirected:
            self.circle_edge_graph.add_edge(v_of_edge, u_of_edge)

    def add_circle_edges_from(self, ebunch_to_add):
        """Add all the edges in ebunch_to_add.

        If you want to add bidirected circle edges, you must pass in
        both (A, B) and (B, A).

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the container will be added to the
            graph. The edges must be given as 2-tuples (u, v) or
            3-tuples (u, v, d) where d is a dictionary containing edge data.

        See Also
        --------
        add_edge : add a single edge
        add_circle_edge : convenient way to add uncertain edges

        Notes
        -----
        Adding the same edge twice has no effect but any edge data
        will be updated when each duplicate edge is added.

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
        >>> e = zip(range(0, 3), range(1, 4))
        >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3

        Associate data to edges

        >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
        >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")
        """
        self._check_adding_edges(ebunch_to_add, EdgeType.circle.value)
        self.circle_edge_graph.add_edges_from(ebunch_to_add)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Override adding edge with check on the PAG."""
        self._check_adding_edge(u_of_edge, v_of_edge, EdgeType.arrow.value)
        return super().add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch, **attr):
        """Override adding multiple edges with check on the PAG."""
        self._check_adding_edges(ebunch, EdgeType.arrow.value)
        return super().add_edges_from(ebunch, **attr)

    def add_bidirected_edge(self, u_of_edge, v_of_edge, **attr) -> None:
        """Override adding bidirected edge with check on the PAG."""
        self._check_adding_edge(u_of_edge, v_of_edge, EdgeType.bidirected.value)
        return super().add_bidirected_edge(u_of_edge, v_of_edge, **attr)

    def add_bidirected_edges_from(self, ebunch, **attr):
        """Override adding bidirected edges with check on the PAG."""
        self._check_adding_edges(ebunch, EdgeType.bidirected.value)
        return super().add_bidirected_edges_from(ebunch, **attr)

    def remove_circle_edge(self, u, v, bidirected: bool = False):
        """Remove circle edge from graph."""
        self.circle_edge_graph.remove_edge(u, v)
        if bidirected:
            self.circle_edge_graph.remove_edge(v, u)

    def has_circle_edge(self, u, v):
        """Check if graph has circle edge from u to v."""
        return self.circle_edge_graph.has_edge(u, v)

    def orient_circle_edge(self, u, v, edge_type: str):
        """Orient circle edge into an arrowhead, or tail.

        Parameters
        ----------
        u : node
            The parent node
        v : node
            The node that 'u' points to in the graph.
        edge_type : str
            An edge type as specified in ``EdgeType``.

        Raises
        ------
        ValueError
            If 'edge_type' is not in the ``EdgeType`` enumeration.
        """
        if edge_type not in EdgeType:
            raise ValueError(
                f"edge_type must be one of {EdgeType}. You passed "
                f"{edge_type} which is unsupported."
            )

        if not self.has_circle_edge(u, v):
            raise RuntimeError(f"There is no circle edge between {u} and {v}.")

        # If there is a circle edge from u -o v, then
        # the subgraph either has u <-o v, or u o-o v
        if self.has_edge(v, u) and edge_type == EdgeType.arrow.value:
            # when we orient (u,v) now as an arrowhead, it is a bidirected arrow
            self.remove_edge(v, u)
            self.remove_circle_edge(u, v)
            self.add_bidirected_edge(u, v)
        elif self.has_edge(v, u) and edge_type == EdgeType.tail.value:
            # when we orient (u,v) now as a tail, we just need to remove the circle edge
            self.remove_circle_edge(u, v)
        elif self.has_circle_edge(v, u) or self.has_circle_edge(u, v):
            # In this case, we have a bidirected circle edge
            # we only need to remove the circle edge and orient
            # it as a normal edge
            self.remove_circle_edge(u, v)
            self.add_edge(u, v)
        else:  # noqa
            raise RuntimeError("The current PAG is invalid.")

    def compute_full_graph(self, to_networkx: bool = False):
        """Compute the full graph from a PAG.

        Adds bidirected edges as latent confounders. Also adds circle edges
        as latent confounders and either:

        - an unobserved mediatior
        - or an unobserved common effect

        The unobserved commone effect will be always conditioned on to
        preserve our notion of m-separation in PAGs.

        Parameters
        ----------
        to_networkx : bool, optional
            Whether to return the graph as a DAG DiGraph, by default False.

        Returns
        -------
        _full_graph : PAG | nx.DiGraph
            The full directed DAG.
        """
        from causal_networkx.utils import _integrate_circle_edges_to_graph

        if self._current_hash != hash(self):
            # first convert latent variables
            explicit_G = super().compute_full_graph()

            # now integrate circle edges
            explicit_G, required_conditioning_set = _integrate_circle_edges_to_graph(explicit_G)

            # update class variables
            self._cond_set = required_conditioning_set
            self._full_graph = explicit_G
            self._current_hash = hash(self)

        if to_networkx:
            return nx.DiGraph(self._full_graph.dag)  # type: ignore
        return self._full_graph

    def edge_type(self, u, v):
        """Return the edge type associated between u and v."""
        if self.has_edge(u, v):
            return EdgeType.arrow.value
        elif self.has_bidirected_edge(u, v):
            return EdgeType.bidirected.value
        elif self.has_circle_edge(u, v):
            return EdgeType.circle.value
        else:
            return None
            # raise RuntimeError(f"Graph does not contain an edge between {u} and {v}.")

    def draw(self):
        """Draw the graph."""
        nx.draw_networkx(self.circle_edge_graph)
        super().draw()

    def is_def_collider(self, node1, node2, node3):
        """Check if <node1, node2, node3> path forms a definite collider.

        I.e. node1 *-> node2 <-* node3.

        Parameters
        ----------
        node1 : node
            A node on the path to check.
        node2 : node
            A node on the path to check.
        node3 : node
            A node on the path to check.

        Returns
        -------
        is_collider : bool
            Whether or not the path is a definite collider.
        """
        # check arrow from node1 into node2
        condition_one = self.has_edge(node1, node2) or self.has_bidirected_edge(node1, node2)

        # check arrow from node2 into node1
        condition_two = self.has_edge(node3, node2) or self.has_bidirected_edge(node3, node2)
        return condition_one and condition_two

    def is_def_noncollider(self, node1, node2, node3):
        """Check if <node1, node2, node3> path forms a definite non-collider.

        I.e. node1 *-* node2 -> node3, or node1 <- node2 *-* node3

        Parameters
        ----------
        node1 : node
            A node on the path to check.
        node2 : node
            A node on the path to check.
        node3 : node
            A node on the path to check.

        Returns
        -------
        is_noncollider : bool
            Whether or not the path is a definite non-collider.
        """
        condition_one = self.has_edge(node2, node3, "arrow") and not self.has_edge(node3, node2)
        condition_two = self.has_edge(node2, node1, "arrow") and not self.has_edge(node1, node2)
        return condition_one or condition_two

    def neighbors(self, u):
        """Get all adjacent nodes of 'u' with any edge to/from it."""
        # we use a dictionary compared to a set to make sure order is kept
        nghbrs = dict()
        for graph in self._graphs:
            graph = graph.to_undirected()
            if u in graph:
                nghbrs.update({node: None for node in graph.neighbors(u)})
        return list(nghbrs.keys())

    def print_edge(self, u, v):
        """Representation of edge between u and v as string.

        Parameters
        ----------
        u : node
            Node in graph.
        v : node
            Node in graph.

        Returns
        -------
        return_str : str
            The type of edge between the two nodes.
        """
        return_str = ""
        if self.has_edge(u, v):
            if self.has_circle_edge(v, u):
                return_str = f"{u} o-> {v}"
            else:
                return_str = f"{u} -> {v}"
        elif self.has_bidirected_edge(u, v):
            return_str = f"{u} <-> {v}"
        elif self.has_circle_edge(u, v):
            if self.has_edge(v, u):
                return_str = f"{u} <-o {v}"
            else:
                return_str = f"{u} o-o {v}"
        return return_str

    def pc_components(self):
        """Possible c-components."""
        pass

    def is_edge_visible(self, u, v):
        """Check if edge (u, v) is visible, or not."""
        pass
        # Given a MAG M , a directed edge A → B in M is visible if there is a
        # vertex C not adjacent to B, such that either there is an edge between
        # C and A that is into A, or there is a collider path between C and A
        # that is into A and every vertex on the path is a parent of B.
        # Otherwise A → B is said to be invisible.

    def to_numpy_array(self) -> NDArray:
        """Convert to a matrix representation.

        A single 2D numpy array is returned, since a PAG only
        maps one edge between any two nodes.

        Returns
        -------
        numpy_graph : np.ndarray of shape (n_nodes, n_nodes)
            The causal graph with values specified as a string
            character. For example, if A has a directed edge to B,
            then the array at indices for A and B has ``'->'``.

        Notes
        -----
        In R's ``pcalg`` package, the following encodes the
        types of edges as an array. We will follow the same encoding
        for our numpy array representation.

            # amat[i,j] = 0 iff no edge btw i,j
            # amat[i,j] = 1 iff i *-o j
            # amat[i,j] = 2 iff i *-> j
            # amat[i,j] = 3 iff i *-- j

        References
        ----------
        https://rdrr.io/cran/pcalg/man/fci.html
        """
        # master list of nodes is in the internal dag
        node_list = self.nodes
        n_nodes = len(node_list)

        numpy_graph = np.zeros((n_nodes, n_nodes))
        bidirected_graph_arr = None
        for name, graph in zip(self._graph_names, self._graphs):
            # make sure all nodes are in the internal graph
            if any(node not in graph for node in node_list):
                graph.add_nodes_from(node_list)

            # handle bidirected edge separately
            if name == EdgeType.bidirected.value:
                bidirected_graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
                continue

            # convert internal graph to a numpy array
            graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
            graph_arr[graph_arr != 0] = PAG_EDGE_MAPPING[name]
            numpy_graph += graph_arr

        if bidirected_graph_arr is not None:
            bidirected_graph_arr[bidirected_graph_arr != 0] = PAG_EDGE_MAPPING[EdgeType.arrow.value]
            numpy_graph += bidirected_graph_arr
        return numpy_graph
