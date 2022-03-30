from enum import Enum
from typing import Set, List

import networkx as nx

from causal_networkx.algorithms.dag import is_directed_acyclic_graph


# TODO: implement graph views for CausalGraph
class CausalGraph:
    """Initialize a causal graphical model.

    This is a causal Bayesian network, where now the edges represent
    causal influences. Self loops are not allowed. This graph type
    inherits functionality from networkx. Two different edge
    types are allowed: bidirected and traditional directed edges.

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

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    networkx.DiGraph
    networkx.Graph

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

    def __init__(
        self, incoming_graph_data=None, incoming_latent_data=None, **attr
    ) -> None:
        # create the DAG of observed variables
        self.dag = nx.DiGraph(incoming_graph_data, **attr)

        # form the bidirected edge graph
        self.c_component_graph = nx.Graph(incoming_latent_data, **attr)

    @property
    def name(self):
        """String identifier of the graph.

        This graph attribute appears in the attribute dict G.graph
        keyed by the string `"name"`. as well as an attribute (technically
        a property) `G.name`. This is entirely user controlled.
        """
        return self.dag.get("name", "")

    @name.setter
    def name(self, s):
        self.dag["name"] = s

    @property
    def c_components(self) -> List[Set]:
        """Generate confounded components of the graph.

        Returns
        -------
        comp : List of sets
            _description_
        """
        return nx.connected_components(self.c_component_graph)

    @property
    def nodes(self):
        return set(self.dag.nodes).union(set(self.c_component_graph.nodes))

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self, u=None, v=None):
        return len(self.dag.number_of_edges(u=u, v=v))

    def number_of_bidirected_edges(self, u=None, v=None):
        return len(self.c_component_graph.number_of_edges(u=u, v=v))

    def has_node(self, n):
        return n in self

    def __str__(self):
        return "".join(
            [
                type(self).__name__,
                f" named {self.name!r}" if self.name else "",
                f" with {self.number_of_nodes()} nodes, ",
                f"{self.number_of_edges()} edges and ",
                f"{self.number_of_bidirected_edges()} bidirected edges"
            ]
        )

    def __contains__(self, n):
        """Returns True if n is a node, False otherwise. Use: 'n in G'.

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
        pass

    def add_node(self, node_for_adding, **attr):
        self.dag.add_node(node_for_adding=node_for_adding, **attr)
        self.c_component_graph.add_node(node_for_adding=node_for_adding, **attr)

    def remove_node(self, n):
        self.dag.remove_node(n)
        self.c_component_graph.remove_node(n)

    def copy(self):
        return CausalGraph(
            self.dag.copy(), self.c_component_graph.copy(), **self.dag.graph
        )

    def clear(self):
        self.dag.clear()
        self.c_component_graph.clear()

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

    def add_bidirected_edge(self, u_of_edge, v_of_edge, **attr) -> None:
        """Add a bidirected edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

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
        self.c_component_graph.add_edge(u_of_edge)

    def children(self, n):
        pass

    def parent(self, n):
        pass

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
        node : nodes
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
        if (not isinstance(dependencies, str)) and \
            (not len(nodes) == len(dependencies)):
                raise ValueError(
                    f'The number of nodes {len(nodes)} should match '
                    f'the number of dependencies {len(dependencies)}.')
        
        assert all(node in self.nodes for node in nodes)

        # create copies of total graph
        bd_graph = self.c_component_graph.copy()
        dag = self.dag.copy()

        for idx, node in enumerate(nodes):
            # remove any bidirected edges incident on nodes, which
            # results in removing the node from the bidirected graph
            bd_graph.remove_node(node)

            if dependencies == 'original':
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
        return is_directed_acyclic_graph(self)

    def subgraph(self, nodes):
        pass

    def edge_subgraph(self, edges):
        pass

    def draw(self):
        """Draws causal graph.

        For custom parametrizations, use ``graphviz``
        or ``networkx`` drawers directly with the
        ``self.dag`` and ``self.c_component_graph``.
        """
        nx.draw_networkx(self.dag)
        nx.draw_networkx(self.c_component_graph, 
            connectionstyle="arc3,rad=-0.4",
            style='dotted')

    def topag(self):
        pass

    def tomag(self):
        pass

    def _classify_three_structure(self, a, b, c):
        """
        Classify three structure as a chain, fork or collider.
        """
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
        pass


class MAG(CausalGraph):
    """Maximal ancestral graph (MAG).

    MAGs only have 1 edge between any 2 nodes. However,
    there are now different types of edges, which represent
    different things. The conditional independences are preserved
    between a causal graph and the associated MAG.

    Undirected edges = networkx.Graph
    Normal directed edges = networkx.DiGraph

    Parameters
    ----------
    CausalGraph : _type_
        _description_

    
    """
    def __init__(self, incoming_graph_data=None, incoming_latent_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, incoming_latent_data, **attr)
    pass


class EdgeType(Enum):
    """Enumeration of different causal edges.

    Categories
    ----------
    arrow : str
        Signifies ">", or "<" edge. That is a normal
        directed edge.
    circle : str
        Signifies "o" endpoint. That is an uncertain edge,
        meaning it could be a tail, or an arrow.

    Notes
    -----
    The possible edges between two nodes thus are:

    ->, <-, <->, o->, <-o, o-o
    """
    arrow = "arrow"
    circle = "circle"


class PAG(nx.DiGraph):
    """Partial ancestral graph (PAG).

    An equivalence class of MAGs, which represent a large
    equivalence class then of causal DAGs. In PAGs, there
    is only one edge between any two nodes, but there are
    different types of edges.

    - "circle": Denotes uncertainty in the edge type. Can be either
    a taile, or an arrow.
    - "arrow": Indicates that all DAGs within this equivalence class
    has an arrow (i.e. "->").

    The entire PAG is represented by one `networkx.DiGraph` with
    edge attributes indicating which type of endpoint the edge has.

    Parameters
    ----------
    CausalGraph : _type_
        _description_
    """
    def __init__(self, incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, **attr)

    def add_edge(self, u_of_edge, v_of_edge):
        """Add a directed edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        See Also
        --------
        add_edges_from : add a collection of edges
        add_uncertain_edge

        Notes
        -----
        Adding an edge that already exists updates the edge data.

        Examples
        --------
        The following all add the edge e=(1, 2) to graph G:

        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = (1, 2)
        >>> G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
        """
        super().add_edge(u_of_edge, v_of_edge, type='arrow')

    def add_uncertain_edge(self, u_of_edge, v_of_edge):
        """Add a directed edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        See Also
        --------
        add_edges_from : add a collection of edges
        add_edge

        """
        super().add_edge(u_of_edge, v_of_edge, type='circle')

    def add_edges_from(self, ebunch_to_add):
        """Add all the edges in ebunch_to_add.

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
        super().add_edges_from(ebunch_to_add, type='arrow')

    def add_uncertain_edges_from(self, ebunch_to_add):
        """Add all the edges in ebunch_to_add.

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
        super().add_edges_from(ebunch_to_add, type='circle')

    def draw(self):
        pass

    def possible_children(self, n):
        pass

    def possible_parents(self, n):
        pass

    def pc_components(self):
        pass

    def cpc_components(self):
        pass

    def is_def_collider(self, node1, node2, node3):
        edge1 = self.get_edge(node1, node2)
        edge2 = self.get_edge(node2, node3)

        if edge1 is None or edge2 is None:
            return False

        return str(edge1.get_proximal_endpoint(node2)) == "ARROW" and str(edge2.get_proximal_endpoint(node2)) == "ARROW"

    # return the endpoint nearest to the given node; returns NoneType if the
    # given node is not along the edge
    def get_proximal_endpoint(self, node: Node) -> Endpoint | None:
        if self.node1 is node:
            return self.endpoint1
        else:
            if self.node2 is node:
                return self.endpoint2
            else:
                return None

    # return the endpoint furthest from the given node; returns NoneType if the
    # given node is not along the edge
    def get_distal_endpoint(self, node: Node) -> Endpoint | None:
        if self.node1 is node:
            return self.endpoint2
        else:
            if self.node2 is node:
                return self.endpoint1
            else:
                return None

    # traverses the edge in an undirected fashion: given one node along the
    # edge, returns the node at the opposite end of the edge
    def get_distal_node(self, node: Node) -> Node | None:
        if self.node1 is node:
            return self.node2
        else:
            if self.node2 is node:
                return self.node1
            else:
                return None

    def is_def_noncollider(self, node1, node2, node3):
        edges = self.get_node_edges(node2)
        circle12 = False
        circle23 = False

        for edge in edges:
            _node1 = edge.get_distal_node(node2) == node1
            _node3 = edge.get_distal_node(node2) == node3

            if _node1 and edge.points_toward(node1):
                return True
            if _node3 and edge.points_toward(node3):
                return True

            if _node1 and edge.get_proximal_endpoint(node2) == Endpoint.CIRCLE:
                circle12 = True
            if _node3 and edge.get_proximal_endpoint(node2) == Endpoint.CIRCLE:
                circle23 = True
            if circle12 and circle23 and not self.is_adjacent_to(node1, node2):
                return True

        return False

    def is_edge_visible(self, u, v):
        pass
        # Given a MAG M , a directed edge A → B in M is visible if there is a vertex C not adjacent to B, such that either there is an edge between C and A that is into A, or there is a collider path between C and A that is into A and every vertex on the path is a parent of B. Otherwise A → B is said to be invisible.