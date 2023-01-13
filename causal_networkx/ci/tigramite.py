from typing import Set, Optional
import pandas as pd
import numpy as np
import networkx as nx
from dodiscover.typing import Column
from dodiscover._protocol import Graph
from dodiscover.ci.base import BaseConditionalIndependenceTest
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.data_processing import DataFrame


class TimeSeriesDiGraph:
    def __init__(self, node_names, max_lag):
        self.G = nx.DiGraph()

        tsnode_names = []
        for t in range(max_lag):
            # each node is for example: A_0, or A_-1
            tsnode_names.extend([f'{node}_{-t}' for node in node_names])

        self.nodes = node_names
        self.max_lag = max_lag
        self.G.add_nodes_from(tsnode_names)

    def add_edge(self, u, v, lag):        
        if lag > 0:
            # add all homologous edges
            self._add_homologous_ts_edges(u, v, lag)
        elif lag == 0:
            # add all homologous contemporaneous edges
            self._add_homologous_contemporaneous_edges(u, v)

    def _add_edge(self, u, v, u_t, v_t):
        self.G.add_edge(f'{u}_{-u_t}', f'{v}_{-v_t}')

    def _add_homologous_contemporaneous_edges(self, u, v):
        for t in range(self.max_lag):
            self._add_edge(u, v, t, t)

    def _add_homologous_ts_edges(self, u, v, lag):
        if lag <= 0:
            raise RuntimeError(f'If lag is {lag}, then add contemporaneous edges.')

        to_t = 0
        for from_t in range(lag, self.max_lag, lag):
            self._add_edge(u, v, from_t, to_t)
            to_t = from_t



class TSOracle(BaseConditionalIndependenceTest):
    """Oracle conditional independence testing for ts graph.

    Used for unit testing and checking intuition.

    Parameters
    ----------
    graph : nx.DiGraph | Graph
        The ground-truth causal graph.
    """

    _allow_multivariate_input: bool = True

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ):
        """Conditional independence test given an oracle.

        Checks conditional independence between 'x_vars' and 'y_vars'
        given 'z_covariates' of variables using the causal graph
        as an oracle. The oracle uses d-separation statements given
        the graph to query conditional independences. This is known
        as the Markov property for graphs
        :footcite:`Pearl_causality_2009,Spirtes1993`.

        Parameters
        ----------
        df : pd.DataFrame of shape (n_samples, n_variables)
            The data matrix. Passed in for API consistency, but not
            used.
        x_vars : node
            A node in the dataset.
        y_vars : node
            A node in the dataset.
        z_covariates : set
            The set of variables to check that separates x_vars and y_vars.

        Returns
        -------
        statistic : None
            A return argument for the statistic.
        pvalue : float
            The pvalue. Return '1.0' if not independent and '0.0'
            if they are.

        References
        ----------
        .. footbibliography::
        """
        self._check_test_input(df, x_vars, y_vars, z_covariates)

        # just check for d-separation between x and y given sep_set
        if isinstance(self.graph, nx.DiGraph):
            is_sep = nx.d_separated(self.graph, x_vars, y_vars, z_covariates)
        else:
            is_sep = nx.m_separated(self.graph, x_vars, y_vars, z_covariates)

        if is_sep:
            pvalue = 1
            test_stat = 0
        else:
            pvalue = 0
            test_stat = np.inf
        return test_stat, pvalue


class TigramiteTSOracle:
    def __init__(self, graph: TimeSeriesDiGraph) -> None:
        self.graph = graph
        self._count = 0

    def set_dataframe(self, dataframe: DataFrame, mask_type = None):
        """Initialize and check the dataframe.

        Parameters
        ----------
        dataframe : data object
            Set tigramite dataframe object. It must have the attributes
            dataframe.values yielding a numpy array of shape (observations T,
            variables N) and optionally a mask of the same shape and a missing
            values flag.

        """
        self.dataframe = dataframe
        if mask_type is not None:
            dataframe._check_mask(require_mask=True)

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max'):
        """Perform conditional independence test.

        Calls the dependence measure and signficicance test functions. The child
        classes must specify a function get_dependence_measure and either or
        both functions get_analytic_significance and  get_shuffle_significance.
        If recycle_residuals is True, also _get_single_residuals must be
        available.

        Parameters
        ----------
        X, Y, Z : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            How many samples to cutoff at the beginning. The default is
            '2xtau_max', which guarantees that MCI tests are all conducted on
            the same samples. For modeling, 'max_lag_or_tau_max' can be used,
            which uses the maximum of tau_max and the conditions, which is
            useful to compare multiple models on the same sample.  Last,
            'max_lag' uses as much samples as possible.

        Returns
        -------
        val, pval : Tuple of floats
            The test statistic value and the p-value.
        """
        self._count += 1
        X_set = set()
        Y_set = set()
        Z_set = set()

        # add X,Y and Z variables in the form that we will feed our CI test
        for (x_var, tau) in X:
            if tau > 0:
                tau = -tau
            X_set.add(f'{x_var}_{tau}')
        for (y_var, tau) in Y:
            if tau > 0:
                tau = -tau
            Y_set.add(f'{y_var}_{tau}')
        for (z_var, tau) in Z:
            if tau > 0:
                tau = -tau
            Z_set.add(f'{z_var}_{tau}')

        if any(node not in self.graph.G.nodes for node in X_set):
            print('Xset not in graph: ', X_set)
            print(self.graph.G.nodes)
        if any(node not in self.graph.G.nodes for node in Y_set):
            print('Yset not in graph: ', Y_set)
            print(self.graph.G.nodes)
        if any(node not in self.graph.G.nodes for node in Z_set):
            print('Zset not in graph: ', Z_set)
            print(self.graph.G.nodes)

        if nx.d_separated(self.graph.G, X_set, Y_set, Z_set):
            val = 0.
            pval = 1.
        else:
            val = 1.
            pval = 0.
        
        return val, pval

def array_to_networkx(arr, node_names):
    """Convert 3D array to networkx graph.

    A "hack" to leverage networkx API, where a

    Parameters
    ----------
    arr : ndarray of shape (n_nodes, n_nodes, max_lag + 1)
        _description_
    node_names : list of nodes
        The node names to apply to the rows and columns.

    Returns
    -------
    G : TimeSeriesDiGraph
        A time-series directed graph.
    """
    n_nodes, _, n_times = arr.shape
    max_lag = int(n_times - 1)

    G = TimeSeriesDiGraph(node_names, max_lag)

    # loop through every node
    for idx in range(n_nodes):
        from_node = node_names[idx]

        for jdx in range(n_nodes):
            to_node = node_names[jdx]
            nz_index = np.argwhere(arr[idx, jdx] != 0).squeeze().astype(int)

            for t in nz_index:
                G.add_edge(from_node, to_node, t)
    
    return G


