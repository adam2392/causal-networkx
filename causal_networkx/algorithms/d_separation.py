import networkx as nx
from networkx.algorithms import d_separated as nx_d_separated

from causal_networkx.cgm import CausalGraph


def convert_latent_to_unobserved_confounders(G: CausalGraph) -> nx.DiGraph:
    """Convert all bidirected edges to unobserved confounders.

    Parameters
    ----------
    G : CausalGraph
        A causal graph with bidirected edges.

    Returns
    -------
    G_copy : nx.DiGraph
        A networkx DiGraph that is a fully specified DAG with unobserved
        variables added in place of bidirected edges.
    """
    uc_label = "Unobserved Confounders"

    G_copy = nx.DiGraph(G.dag)

    # for every bidirected edge, add a new node
    for idx, latent_edge in enumerate(G.c_component_graph.edges):
        G_copy.add_node(f"U{idx}", label=uc_label, observed="no")

        # then add edges from the new UC to the nodes
        G_copy.add_edge("U", latent_edge[0])
        G_copy.add_edge("U", latent_edge[1])
    return G_copy


def d_separated(G: CausalGraph, x, y, z):
    """Check d-separation among 'x' and 'y' given 'z' in graph G.

    This algorithm wraps `networkx.algorithms.d_separated`, but
    allows one to pass in a `CausalGraph` instance instead.

    It first converts all bidirected edges into explicit unobserved
    confounding nodes in an explicit `networkx.DiGraph`, which then
    calls `networkx.algorithms.d_separated` to determin d-separation.
    This inherently increases the runtime cost if there are many
    bidirected edges, because many nodes must be added.

    Parameters
    ----------
    G : CausalGraph
        Causal graph.
    x : set
        First set of nodes in ``G``.
    y : set
        Second set of nodes in ``G``.
    z : set
        Set of conditioning nodes in ``G``. Can be empty set.

    See Also
    --------
    causal_networkx.CausalGraph
    networkx.algorithms.d_separated

    Notes
    -----
    This wraps the networkx implementation, which only allows DAGs. Since
    ``CausalGraph`` is not represented.
    """
    # convert all latent variables to "unobserved" confounders
    explicit_G = convert_latent_to_unobserved_confounders(G)

    # run d-separation
    return nx_d_separated(explicit_G, x, y, z)
