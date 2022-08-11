# Networkx Proposals for Causal graphs and causal graph operations

Networkx is the standard Python package for representing graphs, performing operations on graphs and performing algorithms on graphs. It also enables users to draw graphs. Networkx graphs only have one type of edge, either directed, or undirected.

Causality uses graphs with different types of edges in the same graph. For example:

- ADMG: a causal DAG with directed and bidirected edges.
- CPDAG: an equivalence class of causal graphs with directed and undirected edges.
- PAG: an equivalence class of causal graphs with directed edges, undirected edges and directed edges with circular endpoints.

# MixedEdge Graph Class Extension

We propose a `MixedEdge` graph class that allows arbitrary combinations of existing `Graph` or `DiGraph` classes. 

```Python
class MixedEdgeGraph:
    def __init__(self, graphs: List, edge_types: List):

    def add_edge(self, u, v, edge_type):

    def has_edge(self, u, v, edge_type):
    
    def remove_edge(self, u, v, edge_type):

    def to_undirected(self):
        # convert all graphs to undirected

    @cached_property
    def adj(self, edge_type='all'):
        # return Adjacencies over all/any type of edge
```

This would serve as a "new" base class since it would not inherit from any of the existing networkx graphs, but rather serve
as an API layer for allowing any arbitrary combination of graphs that represent different edge types. Here are a few examples of usage in downstream packages.

```Python
class CPDAG(MixedEdgeGraph):
    def __init__(self, directed_data, undirected_data):
        super().__init__([directed_data, undirected_data], edge_types=['directed', 'undirected'])

    # API layer for users to work with the specific types of edges
    def add_undirected_edge(self, u, v):
        super().add_edge(u, v, edge_type='undirected')

# instantiate the CPDAG
cpdag = CPDAG(nx.DiGraph, nx.Graph)
```

# MixedEdge Graph Algorithms

MixedEdge graphs are useful mainly in causality, so we propose similar to `bipartite` submodule in `networkx` to add a `causal` submodule that has graph traversal algorithms for mixededge graphs. The most relevant one would be that of "m-separation", which generalizes "d-separation".

```Python
def m_separated(G, x, y, z, directed_edge_name='directed', bidirected_edge_name='bidirected'):
    check_G_is_mixed_edge_graph(G)

    # convert G to a DAG by converting all bidirected edges to unobserved confounders
    dag_G = mixed_edge_to_dag(G)

    # run d-separation
    return d_separated(dag_G, x, y, z)
```

Other algorithms also can be proposed, such as discriminating paths, uncovered paths, etc. These are traversal algorithms that operate over specifically mixed edges.

## References

[1] https://github.com/networkx/networkx/discussions/5811