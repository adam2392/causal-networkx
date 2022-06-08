import typing
from typing import Protocol

import networkx as nx
import pandas as pd
from networkx import NetworkXError
from numpy.typing import NDArray

from ..config import EdgeType, EndPoint


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
        self.dag.name = s

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

    @property
    def nodes(self):
        """Return the nodes within the DAG.

        Ignores the c-component graph nodes.
        """
        return self.dag.nodes  # ).union(set(self.c_component_graph.nodes))

    def has_adjacency(self, u, v):
        """Check if there is any edge between u and v."""
        if any(graph.has_edge(u, v) or graph.has_edge(v, u) for graph in self._graphs):
            return True
        return False

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

    def is_directed(self):
        return True

    def is_multigraph(self):
        return False

    def all_edges(self):
        """Get dictionary of all the edges by edge type."""
        return {name: graph.edges for name, graph in zip(self._graph_names, self._graphs)}


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


class AddingEdgeMixin(Protocol):
    def add_chain(self, node_chain):
        """Add a causal chain."""
        ebunch = []
        for idx, node in enumerate(node_chain[:-1]):
            ebunch.append((node, node_chain[idx + 1]))
        self.add_edges_from(ebunch)


class ExportMixin:
    """Mixin class for exporting causal graphs to other formats."""

    @typing.no_type_check
    def to_dot_graph(self, to_dagitty: bool = False) -> str:
        """Convert to 'dot' graph representation as a string.

        The DOT language for graphviz is what is commonly
        used in R's ``dagitty`` package. This is a string
        representation of the graph. However, this converts to a
        string format that is not 100% representative of DOT [1]. See
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

        .. code-block:: text

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

        References
        ----------
        [1] https://github.com/pydot/pydot

        """
        node_str_list = []
        for node in self.nodes:
            node_str_list.append(f"{node};")
        # node_str_list.append(f'{self.nodes[-1]};')
        # node_str = ''.join(node_str_list)
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
                    elif name == EndPoint.circle.value:
                        row = row.replace("->", "-o")
                    edge_list.append(row)
            edge_str_list.extend(edge_list)
        edge_str = "\n".join(edge_str_list)

        # form the final DOT string representation
        if to_dagitty:
            header = "dag"
        else:
            header = "strict digraph"
        dot_graph = header + "\t{\n" f"{node_str}" "\n" f"{edge_str}" "\n" "}"
        return dot_graph

    def to_networkx(self):
        """Converts causal graphs to networkx."""
        from causal_networkx.io import to_networkx

        G = to_networkx(self)
        return G

    def to_pgmpy(self):
        """Convert causal graph to pgmpy."""
        from causal_networkx.io import to_pgmpy

        G = to_pgmpy(self)
        return G

    @typing.no_type_check
    def to_numpy(self) -> NDArray:
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

        References
        ----------
        https://rdrr.io/cran/pcalg/man/fci.html
        """
        from causal_networkx.io import to_numpy

        numpy_graph = to_numpy(self)
        return numpy_graph

    def save(self, fname, format="dot"):
        if format == "dot":
            G = self.to_networkx()
            nx.nx_agraph.write_dot(G, fname)
            # str_dot_graph = self.to_dot_graph()[:-1]
            # graph = pydot.graph_from_dot_data(str_dot_graph)[0]
            # graph.write_dot(fname, encoding="utf-8")
        elif format == "networkx-gml":
            G = self.to_networkx()
            nx.write_gml(G, fname)
        elif format == "pgmpy-bif":
            G = self.to_pgmpy()
            G.save(str(fname), filetype="bif")
        elif format == "txt":
            str_dot_graph = self.to_dot_graph()
            with open(fname, "w") as fout:
                fout.write(str_dot_graph)
