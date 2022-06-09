from typing import Union

import networkx as nx
import numpy as np

from causal_networkx import ADMG, CPDAG, DAG, PAG
from causal_networkx.config import (
    EDGE_TO_VALUE_MAPPING,
    ENDPOINT_TO_EDGE_MAPPING,
    GRAPH_TYPES,
    VALUE_TO_MIXED_EDGE_MAPPING,
    EdgeType,
    EndPoint,
)

GRAPH_TYPE = {DAG: "DAG", ADMG: "ADMG", CPDAG: "CPDAG", PAG: "PAG"}
GRAPH_TYPE_TO_FUNC = {val: key for key, val in GRAPH_TYPE.items()}


def load_from_networkx(G: nx.Graph):
    """Load causal graph from networkx.

    Parameters
    ----------
    G : nx.DiGraph | nx.MultiDiGraph
        The networkx graph, which contains multiple edges if there
        are edge attributes needed for each edge between nodes.
        The edge attributes encode which type of edge it is. See Notes.

    Returns
    -------
    graph : instance of causal DAG
        The causal graph.

    Notes
    -----
    Networkx does not support mixed edge graphs implicitly. However,
    they do support edge attributes. A `networkx.DiGraph` encodes
    a normal causal :class:`causal_networkx.DAG`, while a `networkx.MultiDiGraph` encodes
    all other causal graphs, such as :class:`causal_networkx.CPDAG`,
    :class:`causal_networkx.PAG`, :class:`causal_networkx.ADMG` by storing
    the different edges as edge attributes in the keyword "type".

    Moreover, the graph type is stored in the "graph_type" networkx graph
    attribute.
    """
    graph_func = GRAPH_TYPE_TO_FUNC[G.graph["graph_type"]]
    name = G.name

    graph = graph_func()
    graph.name = name

    # add all nodes to the causal graph
    graph.add_nodes_from(G.nodes)

    # now add all edges
    for u, v, edge_attrs in G.edges.data():
        edge_type = edge_attrs["type"]
        # replace edge marks with their appropriate string representation
        if edge_type == EdgeType.directed.value:
            graph.add_edge(u, v)
        elif edge_type == EdgeType.undirected.value:
            graph.add_undirected_edge(u, v)
        elif edge_type == EdgeType.bidirected.value:
            graph.add_bidirected_edge(u, v)
        elif edge_type == EndPoint.circle.value:
            graph.add_circle_endpoint(u, v)
    return graph


def to_networkx(causal_graph: DAG):
    """Convert causal graph to networkx class.

    Parameters
    ----------
    causal_graph : DAG
        A causal graph.

    Returns
    -------
    G : nx.MultiDiGraph
        The networkx directed graph with multiple edges with edge
        attributes indicating via the keyword "type", which type of
        causal edge it is.
    """
    if len(causal_graph._graphs) == 1:
        G = nx.DiGraph()
    else:
        G = nx.MultiDiGraph()

    # preserve the name
    G.name = causal_graph.name
    graph_type = type(causal_graph).__name__  # GRAPH_TYPE[type(causal_graph)]
    G.graph["graph_type"] = graph_type

    # add all nodes to the networkx graph
    G.add_nodes_from(causal_graph.nodes)

    # add all the edges
    for name, graph in zip(causal_graph._graph_names, causal_graph._graphs):
        # replace edge marks with their appropriate string representation
        if name == EdgeType.directed.value:
            attr = {"type": EdgeType.directed.value}
        elif name == EdgeType.undirected.value:
            attr = {"type": EdgeType.undirected.value}
        elif name == EdgeType.bidirected.value:
            attr = {"type": EdgeType.bidirected.value}
        elif name == EndPoint.circle.value:
            attr = {"type": EndPoint.circle.value}
        G.add_edges_from(graph.edges, **attr)
    return G


def load_from_dot(graph, dagitty: bool = False):
    """Load causal graph from pyDot graph.

    Parameters
    ----------
    graph : _type_
        _description_
    dagitty : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # multiple edges are not allowed
    assert graph.get_strict(None)
    if dagitty:
        assert graph.get_type() == "dag"
    else:
        assert graph.get_type() == "digraph"

    # now read the graph
    N = DAG()

    # assign name of the graph
    name = graph.get_name().strip('"')
    if name != "":
        N.name = name

    # add nodes and attributes
    for p in graph.get_node_list():
        n = p.get_name().strip('"')
        if n in ("node", "graph", "edge"):
            continue
        N.add_node(n, **p.get_attributes())

    # add edges
    for e in graph.get_edge_list():
        u = e.get_source()
        v = e.get_destination()
        attr = e.get_attributes()
        s = []
        d = []
        if isinstance(u, str):
            s.append(u.strip('"'))
        else:
            for unodes in u["nodes"]:
                s.append(unodes.strip('"'))

        if isinstance(v, str):
            d.append(v.strip('"'))
        else:
            for vnodes in v["nodes"]:
                d.append(vnodes.strip('"'))

        for source_node in s:
            for destination_node in d:
                N.add_edge(source_node, destination_node, **attr)

    # add default attributes
    pattr = graph.get_attributes()
    if pattr:
        N.dag["graph"] = pattr
    try:
        N.dag["node"] = graph.get_node_defaults()[0]
    except (IndexError, TypeError):
        pass  # N.graph['node']={}
    try:
        N.dag["edge"] = graph.get_edge_defaults()[0]
    except (IndexError, TypeError):
        pass  # N.graph['edge']={}
    return N


def load_from_pgmpy(pgmpy_dag) -> DAG:
    """Load causal graph from pgmpy.

    Parameters
    ----------
    pgmpy_dag : pgmpy.models.BayesianNetwork
        The Bayesian network from PGMPY.

    Returns
    -------
    dag : DAG
        The causal Bayesian Network.
    """
    adjmat_df = pgmpy_dag["adjmat"]

    # create the causal DAG
    digraph = nx.from_pandas_adjacency(adjmat_df, create_using=nx.DiGraph)
    dag = DAG(digraph)
    return dag


def load_from_numpy(arr, type="dag"):
    """Load causal graph from a numpy array.

    # TODO: add sparse support.

    Parameters
    ----------
    arr : np.ndarray of shape (n_nodes, n_nodes)
        A numpy array specifying the connections between nodes, where
        the ijth component specifies the edge from i to j.
    type : str, optional
        The type of causal graph, by default 'dag'. Must be one of
        ``('dag', ``cpdag``, ``admg``, ``pag``)``. For mixed-edge graphs, the
        ``arr`` specified will have specific values mapped to specific edges.

    Returns
    -------
    G : instance of DAG | CPDAG
        An instance of a causal graph.

    Notes
    -----
    Numpy support for ADMGs are not supported yet, as nodes can have two edges
    between any two nodes (i.e. a directed edge and a bidirected edge).
    """
    n, m = arr.shape
    if n != m:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={arr.shape}")
    if type not in GRAPH_TYPES:
        raise ValueError(
            f'"type" needs to be one of accepted graph types {GRAPH_TYPES}, not {type}.'
        )

    if type == "dag":
        nx_graph = nx.from_numpy_array(arr, create_using=nx.DiGraph)
        G = DAG(nx_graph)
    elif type == "cpdag":
        G = CPDAG()
        # Make sure we get even the isolated nodes of the graph.
        G.add_nodes_from(range(n))

        # Get a list of all the entries in the array with nonzero entries. These
        # coordinates become edges in the graph. (convert to int from np.int64)
        for e in zip(*arr.nonzero()):
            idx = e[0]
            jdx = e[1]

            # get the endpoint value for the ijth connection
            endpoint_ij = VALUE_TO_MIXED_EDGE_MAPPING.get(arr[idx, jdx])

            # check the other endpoint for jith
            endpoint_ji = VALUE_TO_MIXED_EDGE_MAPPING.get(arr[jdx, idx])

            # now map these endpoints to edges that are added to the graph
            edge_type = ENDPOINT_TO_EDGE_MAPPING[(endpoint_ij, endpoint_ji)]

            if edge_type == EdgeType.directed.value:
                # just add directed edge from i to j
                G.add_edge(idx, jdx)
            elif edge_type == EdgeType.undirected.value:
                G.add_undirected_edge(idx, jdx)

    return G


def to_numpy(causal_graph):
    """Convert causal graph to a numpy adjacency array.

    Parameters
    ----------
    causal_graph : instance of DAG
        The causal graph.

    Returns
    -------
    numpy_graph : np.ndarray of shape (n_nodes, n_nodes)
        The numpy array that represents the graph. The values representing edges
        are mapped according to a pre-defined set of values. See Notes.

    Notes
    -----
    The adjacency matrix is defined where the ijth entry of ``numpy_graph`` has a
    non-zero entry if there is an edge from i to j. The ijth entry is symmetric with the
    jith entry if the edge is 'undirected', or 'bidirected'. Then specific edges are
    mapped to the following values:

        - directed edge (->): 1
        - undirected edge (--): 2
        - bidirected edge (<->): 3
        - circle endpoint (-o): 4

    Circle endpoints can be symmetric, but they can also contain a tail, or a directed
    edge at the other end.
    """
    if isinstance(causal_graph, ADMG):
        raise RuntimeError("Converting ADMG to numpy format is not supported.")

    # master list of nodes is in the internal dag
    node_list = causal_graph.nodes
    n_nodes = len(node_list)

    numpy_graph = np.zeros((n_nodes, n_nodes))
    bidirected_graph_arr = None
    graph_map = dict()
    for name, graph in zip(causal_graph._graph_names, causal_graph._graphs):
        # make sure all nodes are in the internal graph
        if any(node not in graph for node in node_list):
            graph = graph.copy()
            graph.add_nodes_from(node_list)

        # handle bidirected edge separately
        if name == EdgeType.bidirected.value:
            bidirected_graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
            continue

        # convert internal graph to a numpy array
        graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
        graph_map[name] = graph_arr

    # ADMGs can have two edges between any 2 nodes
    if type(causal_graph).__name__ == "ADMG":
        # we handle this case separately from the other graphs
        assert len(graph_map) == 1

        # set all bidirected edges with value 10
        bidirected_graph_arr[bidirected_graph_arr != 0] = 10
        numpy_graph += bidirected_graph_arr
        numpy_graph += graph_arr
    else:
        # map each edge to an edge value
        for name, graph_arr in graph_map.items():
            graph_arr[graph_arr != 0] = EDGE_TO_VALUE_MAPPING[name]
            numpy_graph += graph_arr

        # bidirected case is handled separately
        if bidirected_graph_arr is not None:
            numpy_graph += bidirected_graph_arr

    return numpy_graph


def read_dot(fname: str):
    """Read DOT graph from file."""
    import pydot

    if fname.endswith(".dot"):
        graph = pydot.graph_from_dot_file(fname)
    elif fname.endswith(".txt"):
        # read txt file
        with open(fname, "r") as f:
            graph = f.readlines()
        graph = "".join(graph)
        graph = pydot.graph_from_dot_data(graph)

    assert len(graph) == 1
    graph = graph[0]
    nx_graph = nx.drawing.nx_pydot.from_pydot(graph)
    dag = DAG(nx_graph)
    return dag
