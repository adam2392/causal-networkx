from causal_networkx import ADMG, DAG


class TimeSeriesDAG:
    def __init__(self, var_list, max_lag) -> None:
        self._graph = DAG

        self._graph.add_nodes_from(var_list)

    def _add_nodes_over_time(self, nodes, max_lag):
        pass


class TimeSeriesADMG(ADMG):
    # TODO: how to represent causal graph unrolled in time?
    # - an additional graph
    # - edge type indicating time point
    # - maybe do not inherit from ADMG
    def __init__(self, incoming_graph_data=None, incoming_latent_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, incoming_latent_data, **attr)
