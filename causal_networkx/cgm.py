from typing import List, Set

import networkx as nx
import numpy as np
from networkx import NetworkXError

from causal_networkx.config import EdgeType


class NetworkXMixin:
    """Suite of overridden methods of Networkx Graphs.

    Assumes the subclass will store a DiGraph in 'dag' and
    a Graph in 'c_component_graph'.
    """

    @property
    def name(self):
        """Name as a string identifier of the graph.

        This graph attribute appears in the attribute dict G.graph
        keyed by the string `"name"`. as well as an attribute (technically
        a property) `G.name`. This is entirely user controlled.
        """
        return self.dag.name

    @name.setter
    def name(self, s):
        self.dag["name"] = s

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

    def number_of_nodes(self):
        """Return number of nodes in graph."""
        return len(self.nodes)

    def number_of_edges(self, u=None, v=None):
        """Return number of directed edges in graph."""
        return self.dag.number_of_edges(u=u, v=v)

    def number_of_bidirected_edges(self, u=None, v=None):
        """Return number of bidirected edges in graph."""
        return self.c_component_graph.number_of_edges(u=u, v=v)

    def has_node(self, n):
        """Check if graph has node 'n'."""
        return n in self

    def has_edge(self, u, v):
        """Check if graph has edge (u, v)."""
        return self.dag.has_edge(u, v)

    def has_bidirected_edge(self, u, v):
        """Check if graph has bidirected edge (u, v)."""
        return self.c_component_graph.has_edge(u, v)

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
        for graph in self._graphs:
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

    def copy(self):
        """Return a copy of the causal graph."""
        return self.__class__(*self._graphs, **self.dag.graph)

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
        an edge attribute (by default `weight`) to hold a numerical value.

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

    def remove_edges_from(self, ebunch):
        """Remove directed edges."""
        self.dag.remove_edges_from(ebunch)

    def remove_edge(self, u, v):
        """Remove directed edge."""
        self.dag.remove_edge(u, v)

    def add_edges_from(self, ebunch, **attr):
        """Add directed edges."""
        self.dag.add_edges_from(ebunch, **attr)

    def order(self):
        """Return the order of the DiGraph."""
        return self.dag.order()

    def size(self, weight=None):
        """Return the total number of edges possibly with weights."""
        return self.dag.size(weight) + self.c_component_graph.size(weight)

    def degree(self, n):
        """Compute the degree of the DiGraph."""
        return self.dag.degree(n)


class GraphSampleMixin:
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
        # return result_df


# TODO: implement graph views for CausalGraph
class CausalGraph(NetworkXMixin):
    """Initialize a causal graphical model.

    This is a causal Bayesian network, where now the edges represent
    causal influences. Self loops are not allowed. This graph type
    inherits functionality from networkx. Two different edge
    types are allowed: bidirected and traditional directed edges.

    This is also known as an Acyclic Directed Mixed Graph (ADMG).

    Bidirected edges = networkx.Graph
    Normal directed edges = networkx.DiGraph

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize directed acyclic graph. If None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.  If the corresponding optional Python
        packages are installed the data can also be a 2D NumPy array, a
        SciPy sparse matrix, or a PyGraphviz graph. The edges in this graph
        represent directed edges between observed variables, which are
        represented using a `networkx.DiGraph`.

    incoming_latent_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph. The edges in this graph represent
        bidirected edges, which are represented using a `networkx.Graph`.

    incoming_selection_bias : input graph (optional, default: None)
        Data to initialize selection bias graph. Currently,
        not used or implemented.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    networkx.DiGraph
    networkx.Graph

    Subclassing
    -----------
    All causal graphs are a mixture of graphs that represent the different
    types of edges possible. For example, a causal graph consists of two
    types of edges, directed, and bidirected. Each type of edge has the
    following operations:

    - has_<edge_type>_edge: Check if graph has this specific type of edge.
    - add_<edge_type>_edge: Add a specific edge type to the graph.
    - remove_<edge_type>_edge: Remove a specific edge type to the graph.

    Notes
    -----
    The data structure underneath the hood is stored in two networkx graphs:
    `networkx.Graph` and `networkx.DiGraph` to represent the latent unobserved
    confounders and observed variables. These data structures should never be
    modified directly, but should use the CausalGraph class methods.

    Nodes are defined as any nodes defined in the underlying ``DiGraph`` and
    ``Graph``. I.e. Any node connected with either a bidirected, or normal
    directed edge. Adding edges and bidirected edges are performed separately
    in different functions, compared to ``networkx``.
    """

    _graphs: List[nx.Graph]
    _current_hash: int
    _cond_set: Set

    def __init__(
        self,
        incoming_graph_data=None,
        incoming_latent_data=None,
        incoming_selection_bias=None,
        **attr,
    ) -> None:
        # create the DAG of observed variables
        self.dag = nx.DiGraph(incoming_graph_data, **attr)

        # form the bidirected edge graph
        self.c_component_graph = nx.Graph(incoming_latent_data, **attr)

        # form selection bias graph
        # self.selection_bias_graph = nx.Graph(incoming_selection_bias, **attr)

        # keep track of the full graph
        self._full_graph = None
        self._current_hash = None

        # the conditioning set used in d-separation
        # keep track of variables that are always conditioned on
        self._cond_set = set()

        # set the internal graph properties
        self._set_internal_graph_properties()

    def _set_internal_graph_properties(self):
        # create a list of the internal graphs
        self._graphs = [self.dag, self.c_component_graph]

        # number of edges allowed between nodes
        self.allowed_edges = np.inf

    @property
    def bidirected_edges(self):
        """Directed edges."""
        return self.c_component_graph.edges

    def _edge_error_check(self, u_of_edge, v_of_edge):
        pass

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
            return nx.DiGraph(self._full_graph.dag)

        return self._full_graph

    @property
    def c_components(self) -> List[Set]:
        """Generate confounded components of the graph.

        TODO: Improve runtime since this iterates over a list twice.

        Returns
        -------
        comp : List of sets
            The c-components.
        """
        c_comps = nx.connected_components(self.c_component_graph)
        return [comp for comp in c_comps if len(comp) > 1]

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
        nx.MultiDiGraph.add_edges_from : add a collection of edges
        nx.MultiDiGraph.add_edge       : add an edge

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
        nx.MultiDiGraph.add_edges_from : add a collection of edges
        nx.MultiDiGraph.add_edge       : add an edge

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

    def children(self, n):
        """Return an iterator over children of node n.

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

    def do(self, nodes):
        """Apply a do-intervention on nodes to causal graph.

        Parameters
        ----------
        nodes : list of nodes | node
            Either a single node, or list of nodes.

        Returns
        -------
        causal_graph : CausalGraph
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

        return CausalGraph(dag, bd_graph, **self.dag.graph)

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
        causal_graph : CausalGraph
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

        return CausalGraph(dag, bd_graph, **self.dag.graph)

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

    def neighbors(self, u):
        """Get all adjacent nodes of 'u'."""
        pass


# TODO: implement m-separation algorithm
class PAG(CausalGraph):
    """Partial ancestral graph (PAG).

    An equivalence class of MAGs, which represent a large
    equivalence class then of causal DAGs. In PAGs, there
    is only one edge between any two nodes, but there are
    different types of edges.

    The entire PAG is represented using multiple `networkx` graphs:
    - directed edges (normal arrows): `networkx.DiGraph`
    - bidirected edges (indicating latent confounder): `networkx.DiGraph`
    - circular edges (indicating uncertainty in edge type): `networkx.DiGraph`
    - undirected edges (indicating selection bias): `networkx.Graph`. Currently
    not implemented or used.

    Together, these graphs are joined together to form an efficient
    representation of the PAG.

    Parameters
    ----------
    CausalGraph : _type_
        _description_
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
        super().__init__(incoming_graph_data, incoming_latent_data, incoming_selection_data, **attr)

    def _set_internal_graph_properties(self):
        self._graphs = [
            self.dag,
            self.c_component_graph,
            self.circle_edge_graph,
        ]

        # number of allowed edges between any two nodes
        self.allowed_edges = 1

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

    @property
    def circle_edges(self):
        return self.circle_edge_graph.edges

    def number_of_circle_edges(self, u=None, v=None):
        """Return number of bidirected edges in graph."""
        return self.circle_edge_graph.number_of_edges(u=u, v=v)

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
        if not self.dag.has_edge(v_of_edge, u_of_edge) and not bidirected:
            raise RuntimeError(
                f"There is no directed edge from {v_of_edge} to {u_of_edge}. "
                f"Adding a single circle edge is redundant. Are you sure you "
                f"do not intend on adding a bidrected edge?"
            )
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
        add_uncertain_edge : convenient way to add uncertain edges

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
        self.circle_edge_graph.add_edges_from(ebunch_to_add)

    def remove_circle_edge(self, u, v):
        self.circle_edge_graph.remove_edge(u, v)

    def has_circle_edge(self, u, v):
        return self.circle_edge_graph.has_edge(u, v)

    def orient_edge(self, u, v, edge_type: str):
        """Orient an edge a certain way.

        Parameters
        ----------
        u : node
            The parent node
        v : node
            The node that 'u' points to in the graph.
        edge_type : str
            An edge type as specified in `EdgeType`.

        Raises
        ------
        ValueError
            If 'edge_type' is not in the `EdgeType` enumeration.
        """
        if edge_type not in EdgeType:
            raise ValueError(
                f"edge_type must be one of {EdgeType}. You passed "
                f"{edge_type} which is unsupported."
            )
        elif edge_type == EdgeType.arrow.value:
            add_edge_func = self.add_edge
        elif edge_type == EdgeType.circle.value:
            add_edge_func = self.add_circle_edge
        elif edge_type == EdgeType.bidirected.value:
            add_edge_func = self.add_bidirected_edge

        # check what kind of edge it is first and remove it
        if self.has_edge(u, v):
            self.remove_edge(u, v)
        elif self.has_bidirected_edge(u, v):
            self.remove_bidirected_edge(u, v)
        elif self.has_circle_edge(u, v):
            self.remove_circle_edge(u, v)

        add_edge_func(u, v)

    def children(self, n):
        """Return an iterator over children of node n.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        children : Iterator
            An iterator of the children of node 'n'.
        """
        return self.successors(n)

    def parents(self, n):
        """Return an iterator over parents of node n.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        parents : Iterator
            An iterator of the parents of node 'n'.
        """
        return self.predecessors(n)

    def compute_full_graph(self, to_networkx: bool = False):
        """Computes the full graph from a PAG.

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
            return nx.DiGraph(self._full_graph.dag)
        return self._full_graph

    def edge_type(self, u, v):
        """Return the edge type associated between u and v."""
        if self.has_edge(u, v):
            return
        if not self.has_edge(u, v) and not self.has_bidirected_edge(u, v):
            raise RuntimeError(f"Graph does not contain an edge between {u} and {v}.")

        return self[u][v]["type"]

    def draw(self):
        """Draw the graph."""
        pass

    def possible_children(self, n):
        """Possible children of node 'n'."""
        pass

    def possible_parents(self, n):
        """Possible parents of node 'n'."""
        pass

    def pc_components(self):
        """Possible c-components."""
        pass

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
        return self.has_edge(node1, node2, "arrow") and self.has_edge(node3, node2, "arrow")

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

    def is_edge_visible(self, u, v):
        """Check if edge (u, v) is visible, or not."""
        pass
        # Given a MAG M , a directed edge A → B in M is visible if there is a
        # vertex C not adjacent to B, such that either there is an edge between
        # C and A that is into A, or there is a collider path between C and A
        # that is into A and every vertex on the path is a parent of B.
        # Otherwise A → B is said to be invisible.
