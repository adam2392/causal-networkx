from collections import deque

import networkx as nx
import pytest

from causal_networkx import ADMG, DAG
from causal_networkx.algorithms import (
    compute_v_structures,
    is_directed_acyclic_graph,
    topological_sort,
)


# Recipe from the itertools documentation.
def _consume(iterator):
    "Consume the iterator entirely."
    # Feed the entire iterator into a zero-length deque.
    deque(iterator, maxlen=0)


@pytest.mark.parametrize("graph_class", [DAG, ADMG])
class TestDAG:
    @classmethod
    def setup_class(cls):
        pass

    def test_topological_sort1(self, graph_class):
        nx_DG = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
        DG = graph_class(nx_DG)
        assert tuple(topological_sort(DG)) == (1, 2, 3)

        DG.add_edge(3, 2)

        pytest.raises(nx.NetworkXUnfeasible, _consume, topological_sort(DG))

        DG.remove_edge(2, 3)

        assert tuple(topological_sort(DG)) == (1, 3, 2)

        DG.remove_edge(3, 2)

        assert tuple(nx.topological_sort(DG)) in {(1, 2, 3), (1, 3, 2)}

    def test_is_directed_acyclic_graph(self, graph_class):
        G = nx.generators.complete_graph(2)
        with pytest.raises(RuntimeError, match="Causal DAG must be acyclic"):
            G = graph_class(G)
        assert is_directed_acyclic_graph(graph_class(nx.DiGraph([(3, 4), (4, 5)])))


def test_compute_v_structures():
    # build initial DAG
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}, 3: {2: ed2}}
    G = DAG(incoming_graph_data)

    v_structs = compute_v_structures(G)
    assert len(v_structs) == 1
    assert (0, 2, 3) in v_structs

    edges = [("A", "B"), ("C", "B"), ("B", "D"), ("D", "E"), ("G", "E")]
    G = DAG()
    G.add_edges_from(edges)
    v_structs = compute_v_structures(G)
    assert len(v_structs) == 2
