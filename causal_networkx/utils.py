import random
from typing import Callable

import networkx as nx

from causal_networkx.cgm import CausalGraph


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
