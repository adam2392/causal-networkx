import math

import networkx as nx
import numpy as np
import pandas as pd
from manm_cs.graph import Graph
from networkx import DiGraph
from numpy.random import PCG64

from causal_networkx import DAG

ALLOWED_FUNCTIONS = ["linear", "quadratic", "cubic", "tanh", "sin", "cos"]


def lin_func(x):
    return x


def quad_func(x):
    return math.pow(x, 2)


def cube_func(x):
    return math.pow(x, 3)


FUNCTION_DICTIONARY = {
    "linear": lin_func,
    "quadratic": quad_func,
    "cubic": cube_func,
    "tanh": math.tanh,
    "sin": math.sin,
    "cos": math.cos,
}


def simulate_random_graphs_manm_cs(
    n_nodes,
    edge_density,
    n_samples=1000,
    discrete_node_ratio=0.5,
    discrete_signal_to_noise_ratio=0.9,
    min_discrete_value_classes=3,
    max_discrete_value_classes=4,
    continuous_noise_std=1.0,
    with_conditional_gaussian=True,
    functions=[(1.0, lin_func)],
    n_jobs=-1,
    beta_lower_limit=0.5,
    beta_upper_limit=1.0,
    n_latents=None,
    seed=None,
):
    """_summary_

    Parameters
    ----------
    n_nodes : _type_
        Defines the number of nodes to be in the generated DAG.
    edge_density : _type_
        Defines the density of edges in the generated DAG.
    n_samples : int, optional
        Defines the number of samples that shall be generated from the DAG., by default 1000
    discrete_node_ratio : float, optional
        Defines the percentage of nodes that shall be of discrete type. Depending on its
        value the appropriate model (multivariate normal, mixed gaussian, discrete only)
        is chosen, by default 0.5.
    discrete_signal_to_noise_ratio : float, optional
        Defines the probability that no noise is added within the mixed additive noise model,
        by default 0.9.
    min_discrete_value_classes : int, optional
        Defines the minimum number of discrete classes a discrete variable shall have,
        by default 3.
    max_discrete_value_classes : int, optional
        _description_, by default 4
    continuous_noise_std : float, optional
        _description_, by default 1.0
    with_conditional_gaussian : bool, optional
        If True defines that conditional gaussian model is assumed for a mixture of variables.
        Otherwise discrete variables can have continuous parents., by default True
    functions : str, optional
        A list of probabilities and mathematical functions for relationships between two
        continuous nodes. Note, the input are tuples (probability, function), where the
        sum of all probabilities has to equal 1. Command line supported functions are:
        [linear, quadratic, cubic, tanh, sin, cos], by default 'linear'.
    n_jobs : int, optional
        _description_, by default -1
    beta_lower_limit : float, optional
        _description_, by default 0.5
    beta_upper_limit : int, optional
        _description_, by default 1
    n_latents : _type_, optional
        _description_, by default None

    Returns
    -------
    causal_dag : DAG | ADMG
        The causal graph.
    df : pd.DataFrame
        The sampled dataframe.
    """
    from joblib import cpu_count
    from manm_cs.graph import GraphBuilder

    graph_builder = (
        GraphBuilder()
        .with_num_nodes(n_nodes)
        .with_edge_density(edge_density)
        .with_discrete_node_ratio(discrete_node_ratio)
        .with_discrete_signal_to_noise_ratio(discrete_signal_to_noise_ratio)
        .with_min_discrete_value_classes(min_discrete_value_classes)
        .with_max_discrete_value_classes(max_discrete_value_classes)
        .with_continuous_noise_std(continuous_noise_std)
        .with_functions(functions)
        .with_conditional_gaussian(with_conditional_gaussian)
        .with_betas(beta_lower_limit, beta_upper_limit)
    )

    graph = graph_builder.build(seed=seed)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # first sample the data
    df = graph.sample(num_observations=n_samples, num_processes=n_jobs)

    # next let's convert the graph to networkx, so we can convert to DAG
    nx_graph = graph.to_networkx_graph()
    causal_dag = DAG(nx_graph)

    if isinstance(df, list):
        df = pd.concat(df)

    # if we have latents, then let's randomly set certain variables to be latent
    # TODO: MAKE THIS WORK
    return causal_dag, df, graph_builder


def extract_weights(graph: Graph):
    """Assumes manm_cs graph is entirely continuous."""
    test_nx_graph = DiGraph()

    nx_graph = graph.to_networkx_graph()
    top_sort_idx = list(nx.topological_sort(nx_graph))
    variables = graph.variables

    for var in variables:
        # add parents
        parents = var.parents
        betas = var.betas
        var_name = top_sort_idx[var.idx]

        test_nx_graph.add_node(var_name)
        for idx, parent in enumerate(parents):
            parent_name = top_sort_idx[parent.idx]
            weight = betas[idx]

            test_nx_graph.add_node(parent_name)
            test_nx_graph.add_edge(parent_name, var_name, weight=weight)
            nx_graph.add_edge(parent_name, var_name, weight=weight)
    return nx_graph


def _node_funcs():
    pass


def _edge_funcs(num_continuous_parents, random_state=None):
    rng = np.random.Generator(PCG64(seed=random_state))
    functions = []

    # pick randomly from the set of passed in functions
    rand_val = rng.random()
    for function_tuple in sorted(functions):
        if rand_val <= function_tuple[0]:
            return function_tuple[1]

    # use last entry as default
    def identical(value):
        return value

    return identical

    # generate the edge function now for each parent that is continuous
    # functions = [self.chose_function() for p in range(num_continuous_parents)]


def sample(graph, n_samples, random_state=None):
    df = pd.DataFrame()
    for node in graph.nodes:
        df[node] = _sample_node(graph, node, random_state=random_state)

    return df


def _sample_node(graph, node, n_samples, random_state=None):
    # we are at a root node
    if len(graph.parents(node)) == 0:
        signal = np.zeros(n_samples)
    else:
        signal = _compute_signal_from_parents(graph, node, n_samples, random_state=random_state)

    noise = np.random.normal()
    return signal + noise


def _compute_signal_from_parents(graph, node, n_samples, random_state=None):
    pass


def simulate_random_graphs(
    n_nodes,
    edge_density,
    n_samples=1000,
    discrete_node_ratio=0.5,
    discrete_signal_to_noise_ratio=0.9,
    min_discrete_value_classes=3,
    max_discrete_value_classes=4,
    continuous_noise_std=1.0,
    with_conditional_gaussian=True,
    functions=[(1.0, lin_func)],
    n_jobs=-1,
    beta_lower_limit=0.5,
    beta_upper_limit=1.0,
    n_latents=None,
    seed=None,
):
    pass
