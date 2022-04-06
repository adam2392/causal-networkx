from typing import Union
import networkx as nx

from causal_networkx.cgm import CausalGraph
from causal_networkx.algorithms.d_separation import d_separated


class Oracle:
    def __init__(self, graph: Union[CausalGraph, nx.DiGraph]) -> None:
        """Oracle conditional independence testing.

        Used for unit testing and checking intuition.

        Parameters
        ----------
        graph : nx.DiGraph | CausalGraph

        """
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
        else:
            pvalue = 0
        return None, pvalue
