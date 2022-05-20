from typing import Callable, Dict, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd


def learn_skeleton_graph_with_mci(
    X: pd.DataFrame,
    adj_graph: nx.Graph,
    sep_set: Dict[str, Dict[str, Set]],
    parent_dep_dict: Dict[str, Dict[float]],
    ci_estimator: Callable,
    fixed_edges: Optional[Set] = None,
    alpha: float = 0.05,
    max_cond_set_size: int = None,
    **ci_estimator_kwargs
):
    """Perform conditional independence testing using MCI.

    Momentary conditional independence (MCI) is an information-theoretic
    framework proposed in :footcite:`Runge_pcmci_2019` to control for
    false-positives (i.e. incorrect causal arrows) by increasing the
    effect size when performing conditional independence testing.

    Parameters
    ----------
    X : pd.DataFrame
        _description_
    adj_graph : nx.Graph
        _description_
    sep_set : Dict[str, Dict[str, Set]]
        _description_
    parent_dep_dict : Dict[str, Dict[float]]
        _description_
    ci_estimator : Callable
        _description_
    fixed_edges : Optional[Set], optional
        _description_, by default None
    alpha : float, optional
        _description_, by default 0.05
    min_cond_set_size : int, optional
        _description_, by default 0
    max_cond_set_size : int, optional
        _description_, by default None

    Returns
    -------
    adj_graph : nx.Graph
        The more-oriented graph.
    sep_set : set
        The separating set per variable.

    References
    ----------
    .. footbibliography::
    """
    nodes = adj_graph.nodes
    remove_edges = []
    pvalue_of_edges = []

    for i in nodes:
        # iterate through all possible parents of i at this stage
        for j in nodes:
            # if there is no edge
            if not adj_graph.has_edge(i, j):
                continue

            # now we construct the conditioning set
            sep_nodes = set()

            # compute conditional independence test
            test_stat, pvalue = ci_estimator(X, i, j, sep_nodes, **ci_estimator_kwargs)

            # two variables found to be independent given a separating set
            if pvalue > alpha:
                if adj_graph.has_edge(i, j):
                    remove_edges.append((i, j))
                # sep_set[i][j] |= set(cond_set)
                # sep_set[j][i] |= set(cond_set)

    # run FDR method

    # finally remove edges after performing
    # conditional independence tests
    adj_graph.remove_edges_from(remove_edges)
    return adj_graph, sep_set
