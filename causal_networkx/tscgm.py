"""Time-series causal graphs."""

import networkx as nx

from causal_networkx.cgm import ADMG


class TimeSeriesCausalGraph(ADMG):
    def __init__(
        self,
        incoming_graph_data=None,
        incoming_latent_data=None,
        incoming_selection_bias=None,
        **attr,
    ) -> None:
        super().__init__(incoming_graph_data, incoming_latent_data, incoming_selection_bias, **attr)

        # check time series graph is compliant
        self._check_ts_graph()

    def _check_ts_graph(self):
        node_times = nx.get_node_attributes(self.dag, "time")
        if not all(node in node_times for node in self.nodes):
            raise RuntimeError(f"For every node, a time must be " f"specified as a node attribute.")

    def has_edge(self, u, v, t):
        return super().has_edge(u, v)
