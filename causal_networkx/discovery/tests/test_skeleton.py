import networkx as nx
import numpy as np
import pandas as pd
import pytest

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
def test_learn_skeleton(indep_test_func, data_matrix, g_answer, alpha=0.01):
    """Test PC algorithm for estimating the causal DAG."""
    data_df = pd.DataFrame(data_matrix)
    alg = PC(ci_estimator=indep_test_func, alpha=alpha)
    skel_graph, _ = alg.learn_skeleton(data_df)

    # all edges in the answer should be part of the skeleton graph
    for edge in g_answer.edges:
        error_msg = f"Edge {edge} should be in graph {skel_graph}"
        assert skel_graph.has_edge(*edge), error_msg
