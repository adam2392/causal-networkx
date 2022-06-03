from typing import Union

import networkx as nx
import numpy as np

from causal_networkx.algorithms.d_separation import d_separated
from causal_networkx.cgm import ADMG, DAG


class Oracle:
    """Oracle conditional independence testing.

    Used for unit testing and checking intuition.

    Parameters
    ----------
    graph : DAG | ADMG
        The ground-truth causal graph.
    """

    def __init__(self, graph: Union[ADMG, DAG]) -> None:
        self.graph = graph

    def ci_test(self, data, x, y, sep_set):
        """Conditional independence test given an oracle.

        Checks conditional independence between 'x' and 'y'
        given 'sep_set' of variables using the causal graph
        as an oracle.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, n_variables)
            The data matrix. Passed in for API consistency, but not
            used.
        x : node
            A node in the dataset.
        y : node
            A node in the dataset.
        sep_set : set
            The set of variables to check that separates x and y.

        Returns
        -------
        statistic : None
            A return argument for the statistic.
        pvalue : float
            The pvalue. Return '1.0' if not independent and '0.0'
            if they are.
        """
        # just check for d-separation between x and y
        # given sep_set
        is_sep = d_separated(self.graph, x, y, sep_set)

        if is_sep:
            pvalue = 1
            test_stat = 0
        else:
            pvalue = 0
            test_stat = np.inf
        return test_stat, pvalue


class ParentOracle(Oracle):
    """Parent oracle for conditional independence testing.

    An oracle that knows the definite parents of every node.
    """

    def __init__(self, graph: Union[ADMG, DAG]) -> None:
        super().__init__(graph)

    def get_parents(self, x):
        """Return the definite parents of node 'x'."""
        return self.graph.predecessors(x)

    def get_children(self, x):
        """Return the definite children of node 'x'."""
        return self.graph.successors(x)
