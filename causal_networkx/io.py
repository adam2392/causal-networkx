from causal_networkx.cgm import DAG, CPDAG
import networkx as nx

from causal_networkx.config import VALUE_TO_MIXED_EDGE_MAPPING, EdgeType


def load_from_pgmpy(pgmpy_dag):
    adjmat_df = pgmpy_dag["adjmat"]

    # create the causal DAG
    digraph = nx.from_pandas_adjacency(adjmat_df, create_using=nx.DiGraph)
    dag = DAG(digraph)
    return dag


def load_from_numpy(arr, type='dag'):
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
        `arr` specified will have specific values mapped to specific edges.

    Notes
    -----
    """
    n, m = arr.shape
    if n != m:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={arr.shape}")

    if type == 'dag':
        nx_graph = nx.from_numpy_array(arr, create_using=nx.DiGraph)
        graph = DAG(nx_graph)
    elif type == 'cpdag':
        G = CPDAG()
        # Make sure we get even the isolated nodes of the graph.
        G.add_nodes_from(range(n))

        # Get a list of all the entries in the array with nonzero entries. These
        # coordinates become edges in the graph. (convert to int from np.int64)
        directed_edge_graph = []
        undirected_edge_graph = []
        for e in zip(*arr.nonzero()):
            edge = (int(e[0]), int(e[1]))

            endpoint = VALUE_TO_MIXED_EDGE_MAPPING.get(arr[e[0], e[1]])
            # if edge_type == EdgeType.


def load_from_dot(graph, dagitty:bool=False):
    # multiple edges are not allowed
    assert graph.get_strict(None)
    if dagitty:
        assert graph.get_type() == 'dag'
    else:
        assert graph.get_type() == 'digraph'

    # now read the graph
    N = DAG()
    
    # assign name of the graph
    name = graph.get_name().strip('"')
    if name != "":
        N.name = name

    # add nodes and attributes
    for p in graph.get_node_list():
        n = p.get_name().strip('"')
        if n in ('node', 'graph', 'edge'):
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
            for unodes in u['nodes']:
                s.append(unodes.strip('"'))
            
        if isinstance(v, str):
            d.append(v.strip('"'))
        else:
            for vnodes in v['nodes']:
                d.append(vnodes.strip('"'))

        for source_node in s:
            for destination_node in d:
                N.add_edge(source_node, destination_node, **attr)

    # add default attributes
    pattr = graph.get_attributes()
    if pattr:
        N.graph['graph'] = pattr
    try:
        N.graph["node"] = graph.get_node_defaults()[0]
    except (IndexError, TypeError):
        pass  # N.graph['node']={}
    try:
        N.graph["edge"] = graph.get_edge_defaults()[0]
    except (IndexError, TypeError):
        pass  # N.graph['edge']={}
    return N

def read_pgmpy(fname):
    pass

def read_dagitty(fname):
    pass

def read_dot(fname: str):
    import pydot
    if fname.endswith('.dot'):
        graph = pydot.graph_from_dot_file(fname)
    elif fname.endswith('.txt'):
        # read txt file
        with open(fname, "r") as f:
            graph = f.readlines()
        graph = ''.join(graph)
        graph = pydot.graph_from_dot_data(graph)
    
    assert len(graph) == 1
    graph = graph[0]
    nx_graph = nx.drawing.nx_pydot.from_pydot(graph)
    dag = DAG(nx_graph)
    return dag