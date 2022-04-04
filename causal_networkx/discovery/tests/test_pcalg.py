import pytest

import numpy as np
import pandas as pd
import networkx as nx

from causal_networkx.ci import g_square_binary, g_square_discrete
from causal_networkx.ci.tests.testdata import bin_data, dis_data
from causal_networkx.discovery import PC


@pytest.mark.parametrize(
    ("indep_test_func", "data_matrix", "g_answer"),
    [
        (
            g_square_binary,
            np.array(bin_data).reshape((5000, 5)),
            nx.DiGraph(
                {
                    0: (1,),
                    1: (),
                    2: (3, 4),
                    3: (1, 2),
                    4: (1, 2),
                }
            ),
        ),
        (
            g_square_discrete,
            np.array(dis_data).reshape((10000, 5)),
            nx.DiGraph(
                {
                    0: (2,),
                    1: (2, 3),
                    2: (),
                    3: (),
                    4: (3,),
                }
            ),
        ),
    ],
)
def test_estimate_cpdag(indep_test_func, data_matrix, g_answer, alpha=0.01):
    """Test PC algorithm for estimating the causal DAG."""
    data_df = pd.DataFrame(data_matrix)
    alg = PC(ci_estimator=indep_test_func, alpha=alpha)
    alg.fit(data_df)
    graph = alg.graph_

    error_msg = "True edges should be: %s" % (g_answer.edges(),)
    assert nx.is_isomorphic(graph, g_answer), error_msg

    # test what happens if fixed edges are present
    fixed_edges = nx.complete_graph(data_df.columns.values)
    alg = PC(fixed_edges=fixed_edges, ci_estimator=indep_test_func, alpha=alpha)
    alg.fit(data_df)
    complete_graph = alg.graph_
    assert nx.is_isomorphic(complete_graph.to_undirected(), fixed_edges)
    assert not nx.is_isomorphic(complete_graph, g_answer)
