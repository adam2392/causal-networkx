import networkx as nx

from ..config import EdgeType, EndPoint
from .admg import ADMG
from .base import MarkovEquivalenceClass


# TODO: implement m-separation algorithm
class PAG(ADMG, MarkovEquivalenceClass):
    """Partial ancestral graph (PAG).

    An equivalence class of MAGs, which represents an equivalence class of causal DAGs.

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

    incoming_uncertain_data : input graph (optional, default: None)
        Data to initialize circle endpoints on the graph. The edges in this graph
        represent circle endpoints, which are represented using a ``networkx.DiGraph``.
        This does not necessarily need to be acyclic, since there are circle endpoints
        possibly in both directions.

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
    ADMG

    Notes
    -----
    In PAGs, there is only one edge between any two nodes, but there are
    different types of edges. The entire PAG is represented using multiple
    ``networkx`` graphs. Together, these graphs are joined together to form an efficient
    representation of the PAG.

    - directed edges (->, <-, indicating causal relationship): ``networkx.DiGraph``
    - bidirected edges (<->, indicating latent confounder): ``networkx.DiGraph``
    - circular endpoints (-o, o-, indicating uncertainty in edge type): ``networkx.DiGraph``
    - undirected edges (-, indicating selection bias): ``networkx.Graph``. Currently
      not implemented or used.

    Note that the circles are not "edges", but simply endpoints since they can represent
    a variety of different possible endpoints, such as "arrow", or "tail", which then constitute
    a variety of different types of possible edges.

    Compared to causal graphs, PAGs differ in terms of how parents and children
    are defined. In causal graphs, there are only two types of edges, either a
    directed arrow, or bidirected arrow. In PAGs, there are directed arrows with
    either a circle, or tail on the other end: e.g. 'x' o-> 'y'. This now introduces
    "possible" parents/children denoted with a circle edge on the other end of the
    arrow and definite parents/children with only an arrow edge. See `possible_parents`,
    `possible_children` and their counterparts `parents`, `children`.

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
        self.circle_endpoint_graph = nx.DiGraph(incoming_uncertain_data, **attr)

        # construct the causal graph
        super().__init__(
            incoming_graph_data=incoming_graph_data,
            incoming_latent_data=incoming_latent_data,
            incoming_selection_bias=incoming_selection_data,
            **attr,
        )

        # check the PAG
        self._check_pag()

    def _init_graphs(self):
        self._graphs = [
            self.dag,
            self.c_component_graph,
            self.circle_endpoint_graph,
        ]
        self._graph_names = [
            EdgeType.directed.value,
            EdgeType.bidirected.value,
            EndPoint.circle.value,
            # EdgeType.circle.value
        ]

        # number of allowed edges between any two nodes
        self.allowed_edges = 1

    def _check_circle_endpoint(self, node, nghbr):
        raise_error = False
        # check that there is no bidirected arrow in either direction
        if self.has_bidirected_edge(node, nghbr) or self.has_bidirected_edge(nghbr, node):
            raise_error = True

        # check that there is no directed arrow in node -> ngbhr
        elif self.has_edge(node, nghbr):
            raise_error = True

        # check if there is a circle edge in the other direction
        elif self.has_circle_endpoint(nghbr, node):
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
        elif self.has_circle_endpoint(node, nghbr):
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
        if self.has_circle_endpoint(node, nghbr) or self.has_circle_endpoint(nghbr, node):
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
            for nghbr in self.adjacencies(node):
                if self.has_circle_endpoint(node, nghbr) and self.edge_type(node, nghbr).startswith(
                    "circle"
                ):
                    self._check_circle_endpoint(node, nghbr)
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
                f"{self.number_of_circle_endpoints()} circle edges.",
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
            if not self.has_circle_endpoint(n, node):
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
            if not self.has_circle_endpoint(node, n):
                yield node

    @property
    def circle_endpoints(self):
        """Return all circle edges."""
        return self.circle_endpoint_graph.edges

    def number_of_circle_endpoints(self, u=None, v=None):
        """Return number of circle endpoints in graph."""
        return self.circle_endpoint_graph.number_of_edges(u=u, v=v)

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
        if edge_type == EndPoint.circle.value:
            # there should not be an existing arrow
            # nor a bidirected arrow
            if self.has_edge(u_of_edge, v_of_edge) or self.has_bidirected_edge(
                u_of_edge, v_of_edge
            ):
                raise_error = True
        elif edge_type == EdgeType.directed.value:
            # there should not be a circle edge, or a bidirected edge
            if self.has_circle_endpoint(u_of_edge, v_of_edge) or self.has_bidirected_edge(
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

    def add_circle_endpoint(self, u_of_edge, v_of_edge, bidirected: bool = False):
        """Add a circle endpoint between u and v (will add an edge if no previous edge).

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        bidirected : bool
            Whether or not to also add an uncertain endpoint
            from ``v_of_edge`` to ``u_of_edge``.

        See Also
        --------
        add_edges_from : add a collection of edges
        add_edge

        """
        self._check_adding_edge(u_of_edge, v_of_edge, EndPoint.circle.value)

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

        self.circle_endpoint_graph.add_edge(u_of_edge, v_of_edge)
        if bidirected:
            self.circle_endpoint_graph.add_edge(v_of_edge, u_of_edge)

    def add_circle_endpoints_from(self, ebunch_to_add):
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
        add_circle_endpoint : convenient way to add uncertain edges

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
        self._check_adding_edges(ebunch_to_add, EndPoint.circle.value)
        self.circle_endpoint_graph.add_edges_from(ebunch_to_add)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Override adding edge with check on the PAG."""
        self._check_adding_edge(u_of_edge, v_of_edge, EdgeType.directed.value)
        return super().add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch, **attr):
        """Override adding multiple edges with check on the PAG."""
        self._check_adding_edges(ebunch, EdgeType.directed.value)
        return super().add_edges_from(ebunch, **attr)

    def add_bidirected_edge(self, u_of_edge, v_of_edge, **attr) -> None:
        """Override adding bidirected edge with check on the PAG."""
        self._check_adding_edge(u_of_edge, v_of_edge, EdgeType.bidirected.value)
        return super().add_bidirected_edge(u_of_edge, v_of_edge, **attr)

    def add_bidirected_edges_from(self, ebunch, **attr):
        """Override adding bidirected edges with check on the PAG."""
        self._check_adding_edges(ebunch, EdgeType.bidirected.value)
        return super().add_bidirected_edges_from(ebunch, **attr)

    def remove_circle_endpoint(self, u, v, bidirected: bool = False):
        """Remove circle endpoint from graph.

        Removes the endpoint ``u *-o v`` from the graph and orients it
        as ``u *- v``.

        Parameters
        ----------
        u : node
            The start node.
        v : node
            The ending node.
        bidirected : bool, optional
            Whether to also remove the endpoint from v to u, by default False.
        """
        self.circle_endpoint_graph.remove_edge(u, v)
        if bidirected:
            self.circle_endpoint_graph.remove_edge(v, u)

    def has_circle_endpoint(self, u, v):
        """Check if graph has circle endpoint from u to v (``u *-o v``)."""
        return self.circle_endpoint_graph.has_edge(u, v)

    def orient_circle_endpoint(self, u, v, endpoint: str):
        """Orient circle endpoint into an arrowhead, or tail.

        Parameters
        ----------
        u : node
            The parent node
        v : node
            The node that 'u' points to in the graph.
        endpoint : str
            An edge type as specified in ``EndPoint`` ('arrow', 'tail')

        Raises
        ------
        ValueError
            If 'endpoint' is not in the ``EndPoint`` enumeration.
        """
        if endpoint not in EndPoint:
            raise ValueError(
                f"endpoint must be one of {EndPoint}. You passed "
                f"{endpoint} which is unsupported."
            )

        if not self.has_circle_endpoint(u, v):
            raise RuntimeError(f"There is no circle endpoint between {u} and {v}.")

        # Performs orientation of edges
        if self.has_edge(v, u) and endpoint == EndPoint.arrow.value:
            # Orients: u <-o v => u <-> v
            # when we orient (u,v) now as an arrowhead, it is a bidirected arrow
            self.remove_edge(v, u)
            self.remove_circle_endpoint(u, v)
            self.add_bidirected_edge(u, v)
        elif self.has_edge(v, u) and endpoint == EndPoint.tail.value:
            # Orients: u <-o v => u <- v
            # when we orient (u,v) now as a tail, we just need to remove the circle edge
            self.remove_circle_endpoint(u, v)
        elif self.has_circle_endpoint(v, u) and endpoint == EndPoint.arrow.value:
            # Orients: u o-o v => u o-> v
            # In this case, we have a bidirected circle edge
            # we only need to remove the circle edge and orient
            # it as a normal edge
            self.remove_circle_endpoint(u, v)
            self.add_edge(u, v)
        elif self.has_circle_endpoint(v, u) and endpoint == EndPoint.tail.value:
            # Orients: u o-o v => u o-- v
            raise RuntimeError("Selection bias has not been implemented into PAGs yet.")
            # In this case, we have a bidirected circle edge
            # we only need to remove the circle edge and orient
            # it as possibly a directed, or undirected edge
            self.remove_circle_endpoint(u, v)
            self.add_edge(u, v)
        elif self.has_circle_endpoint(u, v) and endpoint == EndPoint.arrow.value:
            # In this case, we have a circle edge that is oriented into an arrowhead
            # we only need to remove the circle edge and orient
            # it as a normal edge
            self.remove_circle_endpoint(u, v)
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
        from causal_networkx.utils import _integrate_circle_endpoints_to_graph

        if self._current_hash != hash(self):
            # first convert latent variables
            explicit_G = super().compute_full_graph()

            # now integrate circle edges
            explicit_G, required_conditioning_set = _integrate_circle_endpoints_to_graph(explicit_G)

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
            return EdgeType.directed.value
        elif self.has_bidirected_edge(u, v):
            return EdgeType.bidirected.value
        elif self.has_circle_endpoint(u, v):
            if self.has_edge(v, u):
                return EdgeType.circle_to_directed.value
            elif self.has_circle_endpoint(v, u):
                return EdgeType.circle_to_circle.value
            # TODO: add undirected edge possibility for selection bias
        else:
            return None

    def draw(self):
        """Draw the graph."""
        nx.draw_networkx(self.circle_endpoint_graph)
        super().draw()

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

    def is_edge_visible(self, u, v):
        """Check if edge (u, v) is visible, or not."""
        pass
        # Given a MAG M , a directed edge A → B in M is visible if there is a
        # vertex C not adjacent to B, such that either there is an edge between
        # C and A that is into A, or there is a collider path between C and A
        # that is into A and every vertex on the path is a parent of B.
        # Otherwise A → B is said to be invisible.

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
            if self.has_circle_endpoint(v, u):
                return_str = f"{u} o-> {v}"
            else:
                return_str = f"{u} -> {v}"
        elif self.has_bidirected_edge(u, v):
            return_str = f"{u} <-> {v}"
        elif self.has_circle_endpoint(u, v):
            if self.has_edge(v, u):
                return_str = f"{u} <-o {v}"
            else:
                return_str = f"{u} o-o {v}"
        return return_str
