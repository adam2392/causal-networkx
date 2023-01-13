import networkx as nx 
import numpy as np
import pywhy_graphs
from collections import defaultdict
from pywhy_graphs.array import api
from pywhy_graphs.simulate import simulate_var_process_from_summary_graph

from dodiscover import FCI, PC
from dodiscover.ci import Oracle
from dodiscover import make_context
from pywhy_graphs.viz import draw

import tigramite
from tigramite.independence_tests import OracleCI
import tigramite.data_processing as pp
from tigramite.lpcmci import LPCMCI

from causal_networkx.ci.tigramite import array_to_networkx, TigramiteTSOracle


def tig():
    # Set a seed for reproducibility
    seed = 19

    # Choose the time series length
    T = 500

    # Specify the model (note that here, unlike in the typed equations, variables
    # are indexed starting from 0)
    def lin(x): return x
    from tigramite.toymodels import structural_causal_processes as toys

    links = {0: [((0, -1), 0.9, lin), ((1, 0), 0.6, lin)],
            1: [],
            2: [((2, -1), 0.9, lin), ((1, -1), 0.4, lin)],
            3: [((3, -1), 0.9, lin), ((2, -2), -0.5, lin)]                                    
            }

    # Specify dynamical noise term distributions, here unit variance Gaussians
    random_state = np.random.RandomState(seed)
    noises = noises = [random_state.randn for j in links.keys()]

    # Generate data according to the full structural causal process
    data_full, nonstationarity_indicator = toys.structural_causal_process(
        links=links, T=T, noises=noises, seed=seed)
    assert not nonstationarity_indicator

    # Remove the unobserved component time series
    data_obs = data_full[:, [0, 2, 3]]

    # Number of observed variables
    N = data_obs.shape[1]

    # Initialize dataframe object, specify variable names
    var_names = [r'$X^{%d}$' % j for j in range(N)]
    dataframe = pp.DataFrame(data_obs, var_names=var_names)
    print(dataframe.values.keys())
    for df in dataframe.values.values():
        print(df.shape)

def get_all_missing_edges(G: nx.MixedEdgeGraph):
    nodes = G.nodes
    missing_edges = []
    for node in nodes:
        for check_node in nodes:
            if node == check_node:
                continue

            if check_node not in G.neighbors(node):
                missing_edges.append((node, check_node))
    return missing_edges


def main():
    max_lag = 2
    n_times = 100
    random_state=12345

    # define a summary graph
    directed_edges = nx.DiGraph(
        [
            ("x8", "x2"),
            ("x9", "x2"),
            ("x10", "x1"),
            ("x2", "x4"),
            ("x4", "x6"),  # start of cycle
            ("x6", "x5"),
            ("x5", "x3"),
            ("x3", "x4"),  # end of cycle
            ("x6", "x7"),
        ]
    )
    bidirected_edges = nx.Graph([("x1", "x3")])
    G = nx.MixedEdgeGraph([directed_edges, bidirected_edges], ["directed", "bidirected"])
    n_nodes = G.number_of_nodes()

    # use the acyclified graph
    acy_G = pywhy_graphs.acyclification(G)

    # generate data for VAR
    ts_arr, data = simulate_var_process_from_summary_graph(
        G, max_lag=max_lag, n_times=n_times, random_state=random_state)
    ts_arr = np.concatenate((np.zeros((n_nodes, n_nodes, 1)), ts_arr), axis=-1)
    arr_idx = np.array(G.nodes)
    # arr_mapping = {idx: node for idx, node in enumerate(arr_idx)}
    # # convert time-series graph to a dictionary of lagged links
    # lagged_links = api.array_to_lagged_links(
    #         ts_arr, arr_idx=np.arange(n_nodes),
    #         include_weights=True)

    # Naive run to learn ts-full graph:
    # Here, we use an oracle for the time-series graph to
    # count the number of tests we will do.
    # convert to 3D networkx graph
    ts_nx_graph = array_to_networkx(ts_arr, np.arange(n_nodes))
    ts_oracle_ci = TigramiteTSOracle(graph=ts_nx_graph)

    # convert to Tigramite DF
    tig_df = pp.DataFrame(data=data.to_numpy(),
        var_names=arr_idx.tolist()
     )
    # tig_ci_test = OracleCI(links=lagged_links, tau_max=max_lag, 
    #     observed_vars=list(lagged_links.keys())
    #     )
    ts_alg = LPCMCI(tig_df, cond_ind_test=ts_oracle_ci, verbosity=0)
    ts_alg.run_lpcmci(tau_max=max_lag-1)
    print(ts_oracle_ci._count)

    # Intermediate knowledge run to learn full ts-graph:
    # Informed run, where we will first learn the PAG of
    # the summary graph, and create a rule to trim variables

    # use oracle to run FCI algorithm
    ci_test = Oracle(acy_G)
    context = make_context().variables(data=data).build()

    # run FCI algorithm on the summary graph and keep track
    # of the number of CI tests that were run
    alg = FCI(ci_estimator=ci_test)
    alg.fit(data, context)
    n_ci_tests = alg.n_ci_tests
    graph = alg.graph_
    print(n_ci_tests)

    # At this point, we have the resulting PAG for the
    # cyclic summary graph.
    # we will:
    # - for all nodes find all other nodes w/o an adjacency
    # connection w/ this node
    # - remove all these edges

    # get the variables that are definitely d-separated
    def_no_edge = set()

    for idx, node in enumerate(graph.nodes):
        for jdx, other_node in enumerate(graph.nodes):
            if node == other_node:
                continue

            if other_node not in graph.neighbors(node):
                def_no_edge.add(frozenset([idx, jdx]))

    print(f'Definitely trimmed {def_no_edge}...')

    # map all node names to relevant indices which are used in LPCMCI
    num_nodes = acy_G.number_of_nodes()
    min_lag = 0
    max_lag = 1
    selected_links = {j: [(i, -tau) for i in range(num_nodes) for tau in range(min_lag, max_lag + 1) if (tau > 0 or j != i)] for j in range(num_nodes)}

    # now trim definite no edges
    trimmed_links = defaultdict(list)
    for idx, links in selected_links.items():
        for link in links:
            if (idx, link[0]) not in def_no_edge:
                trimmed_links[idx].append(link)
    
    ts_oracle_ci = TigramiteTSOracle(graph=ts_nx_graph)
    ts_alg = LPCMCI(tig_df, cond_ind_test=ts_oracle_ci, verbosity=0)
    ts_alg.run_lpcmci(selected_links=trimmed_links, tau_max=max_lag)
    print(ts_oracle_ci._count)
    print(ts_oracle_ci._count - ts_alg.n_saved_tests)


def main_simple():
    max_lag = 2
    n_times = 100
    random_state=12345

    # define a summary graph
    directed_edges = nx.DiGraph(
        [
            ('a', 'y'),
            ('y', 'b'),
            ('y', 'x'), # cycle start
            ('x', 'y'),
            # ('w', 'x'), 
            # ('x', 'y'), # cycle end
            ('x', 'z'),
            ('y', 'z'),
        ]
    )
    G = nx.MixedEdgeGraph([directed_edges], ["directed"])
    n_nodes = G.number_of_nodes()

    # use the acyclified graph
    acy_G = pywhy_graphs.acyclification(G)

    # generate data for VAR
    ts_arr, data = simulate_var_process_from_summary_graph(
        G, max_lag=max_lag, n_times=n_times, random_state=random_state)
    ts_arr = np.concatenate((np.zeros((n_nodes, n_nodes, 1)), ts_arr), axis=-1)
    arr_idx = np.array(G.nodes)
    arr_mapping = {idx: node for idx, node in enumerate(arr_idx)}
    # data.columns = np.arange(n_nodes)

    # use oracle to run FCI algorithm
    ci_test = Oracle(acy_G)
    context = make_context().variables(data=data).build()

    # run FCI algorithm on the summary graph and keep track
    # of the number of CI tests that were run
    alg = FCI(ci_estimator=ci_test)
    alg.fit(data, context)
    n_ci_tests = alg.n_ci_tests
    graph = alg.graph_

    # get the variables that are definitely d-separated
    def_no_edge = []
    for node in graph.nodes:
        for other_node in graph.nodes:
            if node == other_node:
                continue

            if other_node not in graph.neighbors(node):
                def_no_edge.append((node, other_node))

    # construct preliminary knowledge

    dot_graph = draw(graph, directed_graph_name='directed')
    print(n_ci_tests)
    dot_graph.render(outfile="oracle_pag.png", view=True)
    print(acy_G.edges())
    dot_graph = draw(acy_G, directed_graph_name='directed')
    dot_graph.render(outfile="acy_G.png", view=True)

    print(G.edges())
    dot_graph = draw(G, directed_graph_name='directed')
    dot_graph.render(outfile="G.png", view=True)


if __name__ == '__main__':
    # tig()
    main()
    # main_simple()