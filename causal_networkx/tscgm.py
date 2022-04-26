"""Time-series causal graphs."""


from causal_networkx.cgm import CausalGraph


class TimeSeriesCausalGraph(CausalGraph):
    def __init__(
        self,
        incoming_graph_data=None,
        incoming_latent_data=None,
        incoming_selection_bias=None,
        **attr
    ) -> None:
        super().__init__(incoming_graph_data, incoming_latent_data, incoming_selection_bias, **attr)
