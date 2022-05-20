from causal_networkx.cgm import ADMG


class TimeSeriesADMG(ADMG):
    # TODO: how to represent causal graph unrolled in time?
    # - an additional graph
    # - edge type indicating time point
    # - maybe do not inherit from ADMG
    def __init__(self, incoming_graph_data=None, incoming_latent_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, incoming_latent_data, **attr)
