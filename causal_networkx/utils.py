from typing import Callable
import networkx as nx

from causal_networkx.cgm import CausalGraph


def undirected_to_pag():
    pass


def _check_ci_estimator(ci_estimator: Callable):
    pass


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
