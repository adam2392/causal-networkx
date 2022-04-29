import random
from typing import Callable

import networkx as nx

from causal_networkx.cgm import PAG, CausalGraph


def _check_ci_estimator(ci_estimator: Callable):
    pass


def _sample_cg(
    n, dir_rate, bidir_rate, enforce_direct_path=False, enforce_bidirect_path=False, enforce_ID=None
):
    """Sample a random causal diagram with n variables, including X and Y.

    Parameters
    ----------
    n : int
        Number of variables in the graph.
    dir_rate : float
        All directed edges are independently included with a
        chance of dir_rate.
    bidir_rate : float
        All bidirected edges are independently included with a
        chance of bidir_rate, which is between 0 and 1.
    enforce_direct_path : bool, optional
        If True, then there is guaranteed to be a directed path from X to Y
        this implies almost surely that P(Y | do(X)) != P(Y), by default False.
    enforce_bidirect_path : bool, optional
        If True, then there is guaranteed to be a bidirected path from X to Y
        this implies P(Y | do(X)) is not amenable to backdoor adjustment, by
        default False.
    enforce_ID : bool, optional
        If True, then P(Y | do(X)) is guaranteed to be identifiable.
        If False, then P(Y | do(X)) is guaranteed to not be identifiable,
        by default None.

    Returns
    -------
    cg : CausalGraph
        Sampled causal graph.
    """
    cg = None
    done = False

    while not done:
        x_loc = random.randint(0, n - 2)
        V_list = ["V{}".format(i + 1) for i in range(n - 2)]
        V_list.insert(x_loc, "X")
        V_list.append("Y")

        de_list = []
        be_list = []
        for i in range(len(V_list) - 1):
            for j in range(i + 1, len(V_list)):
                if random.random() < dir_rate:
                    de_list.append((V_list[i], V_list[j]))
                if random.random() < bidir_rate:
                    be_list.append((V_list[i], V_list[j]))

        cg = CausalGraph(V_list, de_list, be_list)

        done = True
        # if enforce_direct_path and not graph_search(cg, "X", "Y", edge_type="direct"):
        #     done = False
        # if enforce_bidirect_path and not graph_search(cg, "X", "Y", edge_type="bidirect"):
        #     done = False

        # if enforce_ID is not None:
        #     id_status = identify(X={"X"}, Y={"Y"}, G=cg) != "FAIL"
        #     if enforce_ID != id_status:
        #         done = False

    return cg


def convert_latent_to_unobserved_confounders(G: CausalGraph) -> CausalGraph:
    """Convert all bidirected edges to unobserved confounders.

    Parameters
    ----------
    G : CausalGraph
        A causal graph with bidirected edges.

    Returns
    -------
    G_copy : CausalGraph
        A networkx DiGraph that is a fully specified DAG with unobserved
        variables added in place of bidirected edges.
    """
    uc_label = "Unobserved Confounders"
    G_copy = G.copy()

    # for every bidirected edge, add a new node
    for idx, latent_edge in enumerate(G.c_component_graph.edges):
        G_copy.add_node(f"U{idx}", label=uc_label, observed="no")

        # then add edges from the new UC to the nodes
        G_copy.add_edge(f"U{idx}", latent_edge[0])
        G_copy.add_edge(f"U{idx}", latent_edge[1])

        # remove the actual bidirected edge
        G_copy.remove_bidirected_edge(*latent_edge)

    return G_copy


def _integrate_circle_edges_to_graph(G: PAG):
    """Add circle edges into a graph.

    Represents circle edges using additional nodes that are added
    into the graph, to "preserve" m-separation properties. Since
    a circle edges represents uncertainty about the edge type, we
    will add all edges possible regarding the uncertainty.

    Parameters
    ----------
    G : PAG
        The PAG.

    Returns
    -------
    G_copy : CausalGraph
        The causal graph with the modified edges.

    Notes
    -----
    Even though A <-o B can be A <- B, or A <- UC -> B

    The circle edges that are possible can be represented in a
    full graph as:

    - A <-o B is turned to A <- UC <- B, and A <- UE -> B since that
    the edges restrict to A <-* B, preserving a possible collider
    at node A
    - A o-o B is turned to A -> UC -> B and A <- UE <- B. In this case,
    we want to preserve a possible collider at both nodes A and B.

    where unobserved confounders (UC), or unobserved common effects (UE)
    are added to the graph.
    """
    if not isinstance(G, PAG):
        raise ValueError(f"Graph {G} should be a PAG.")

    G_copy = G.copy()
    required_conditioning_set = set()

    # for every bidirected edge, add a new node
    for idx, circle_edge in enumerate(G.circle_edge_graph.edges):
        # check if there is a bidirected circle edge
        if G.has_circle_edge(circle_edge[1], circle_edge[0]):
            # create unobserved confounder
            G_copy.add_edge(f"U{idx}", circle_edge[0])
            G_copy.add_edge(f"U{idx}", circle_edge[1])

            # create an "unobserved" common effect
            # that will always be conditioned on
            # then add edges from the nodes to the new UE
            G_copy.add_edge(circle_edge[0], f"UE{idx}")
            G_copy.add_edge(circle_edge[1], f"UE{idx}")
            required_conditioning_set.add(f"UE{idx}")
        else:
            # there is a <-o, or o-> edge between these two nodes
            # create a path from A -> uc -> B
            G_copy.add_edge(circle_edge[0], f"uma{idx}")
            G_copy.add_edge(f"uma{idx}", circle_edge[1])

            # now add an unobserved confounder
            G_copy.add_edge(f"umb{idx}", circle_edge[0])
            G_copy.add_edge(f"umb{idx}", circle_edge[1])

        G_copy.remove_circle_edge(*circle_edge)

    return G_copy, required_conditioning_set


# TODO: integrat into causal graph
def convert_selection_vars_to_common_effects(G: CausalGraph) -> nx.DiGraph:
    """Convert all undirected edges to unobserved common effects.

    Parameters
    ----------
    G : CausalGraph
        A causal graph with undirected edges.

    Returns
    -------
    G_copy : CausalGraph
        A causal graph that is a fully specified DAG with unobserved
        selection variables added in place of undirected edges.
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
